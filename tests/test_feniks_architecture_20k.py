from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from euclid_dsps.amortized.config import amortized_config
from euclid_dsps.amortized.elbo import objective_uses_truth
from euclid_dsps.amortized.flows import StandardNormalPrior
from euclid_dsps.amortized.posterior import (
    ConditionalFlowEncoder,
    posterior_log_prob,
    sample_posterior,
)
from euclid_dsps.config import load_config
from scripts.build_feniks_architecture_20k_manifests import build
from scripts.select_feniks_architecture_20k import LABELS, select

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
CONFIGS = (
    "feniks_architecture_20k_current_realnvp.yaml",
    "feniks_architecture_20k_set_realnvp.yaml",
    "feniks_architecture_20k_set_autoregressive_spline.yaml",
)


def test_autoregressive_spline_has_exact_normalized_density() -> None:
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(40),
        input_dim=6,
        latent_dim=3,
        hidden_sizes=(12, 12),
        activation="gelu",
        log_std_min=-6.0,
        log_std_max=2.0,
        initial_log_std=-1.0,
        family="autoregressive_rq_spline",
        n_layers=2,
        hidden_size=12,
        n_bins=6,
        output_space="latent_x",
    )
    model = type("Model", (), {})()
    model.encoder = encoder
    model.prior = StandardNormalPrior(latent_dim=3)
    features = jax.random.normal(jax.random.PRNGKey(41), (4, 6))

    posterior = sample_posterior(
        model,
        jax.random.PRNGKey(42),
        features,
        5,
    )
    evaluated = jax.vmap(lambda value: posterior_log_prob(model, features, value))(
        posterior.x
    )

    assert posterior.x.shape == (5, 4, 3)
    assert jnp.all(jnp.isfinite(posterior.logq))
    assert jnp.allclose(evaluated, posterior.logq, atol=5.0e-4)


def test_set_context_is_direct_and_differentiable() -> None:
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(43),
        input_dim=8,
        latent_dim=3,
        hidden_sizes=(12, 12),
        activation="gelu",
        log_std_min=-6.0,
        log_std_max=2.0,
        initial_log_std=-1.0,
        family="realnvp",
        n_layers=2,
        hidden_size=12,
        output_space="latent_x",
        context_encoder_type="passband_set_transformer",
        set_n_bands=4,
        set_token_dim=12,
        set_context_dim=10,
        set_num_heads=3,
        set_num_layers=2,
    )
    features = jax.random.normal(jax.random.PRNGKey(44), (2, 8))
    mean, log_std = encoder(features)
    context = encoder.flow_context(features, mean, log_std)
    gradient = jax.grad(lambda value: jnp.sum(encoder.flow_context(value)))(features[0])

    assert mean.shape == (2, 3)
    assert log_std.shape == (2, 3)
    assert context.shape == (2, 10)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 0.0)


@pytest.mark.parametrize("config_name", CONFIGS)
def test_architecture_configs_share_self_supervised_frozen_prior_contract(
    config_name: str,
) -> None:
    cfg = amortized_config(load_config(CONFIG_DIR / config_name))

    assert cfg["prior"]["train_jointly"] is False
    assert cfg["objective"]["wake"]["train_prior"] is False
    assert cfg["objective"]["prior_truth_weight"] == 0.0
    assert not objective_uses_truth(cfg["objective"])
    assert cfg["likelihood"]["type"] == "student_t"
    assert cfg["likelihood"]["student_t_dof"] == 2.0
    assert cfg["encoder"]["flow_output_space"] == "latent_x"
    assert cfg["training"]["epochs"] == 80


def test_architecture_matrix_changes_one_axis_at_a_time() -> None:
    current, set_coupling, set_autoregressive = [
        amortized_config(load_config(CONFIG_DIR / name)) for name in CONFIGS
    ]

    assert current["encoder"]["context_encoder"] == "base_moments"
    assert current["encoder"]["flow_family"] == "realnvp"
    assert set_coupling["encoder"]["context_encoder"] == ("passband_set_transformer")
    assert set_coupling["encoder"]["flow_family"] == "realnvp"
    assert (
        set_autoregressive["encoder"]["context_encoder"]
        == (set_coupling["encoder"]["context_encoder"])
    )
    assert set_autoregressive["encoder"]["flow_family"] == ("autoregressive_rq_spline")


def test_manifest_builder_uses_disjoint_row_identities_only(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    pq.write_table(pa.table({"flux": np.arange(40)}), train)
    pq.write_table(pa.table({"flux": np.arange(20)}), test)
    out = tmp_path / "manifests"

    payload = build(
        train_catalog=train,
        test_catalog=test,
        out=out,
        n_train=18,
        n_validation=2,
        n_probe=5,
        seed=7,
    )
    train_indices = np.load(out / "train_indices.npy")
    validation_indices = np.load(out / "validation_indices.npy")
    probe_indices = np.load(out / "blind_iw_probe_indices.npy")

    assert len(np.intersect1d(train_indices, validation_indices)) == 0
    assert len(train_indices) == 18
    assert len(validation_indices) == 2
    assert len(probe_indices) == 5
    assert "no truth column is read" in payload["contract"].lower()


def test_selector_never_promotes_a_failed_support_candidate(tmp_path: Path) -> None:
    seeds = (1, 2)
    for label_index, label in enumerate(LABELS):
        for seed in seeds:
            out = tmp_path / label / f"seed_{seed}"
            out.mkdir(parents=True)
            payload = {
                "candidate": label,
                "seed": seed,
                "ordinary_iw_support": "FAIL",
                "median_raw_ess_fraction": 0.01 + 0.001 * label_index,
                "fraction_pareto_k_gt_0p7": 0.6,
                "best_validation_objective": 12.0,
            }
            (out / "candidate_summary.json").write_text(json.dumps(payload))

    result = select(tmp_path, seeds)

    assert result["selection_status"] == "DIAGNOSTIC_ONLY"
    assert result["selected_architecture"] is None
    assert result["ready_for_selected_feniks_adaptation"] is False


def test_slurm_wrapper_encodes_six_runs_and_truth_free_selection() -> None:
    worker = (ROOT / "scripts" / "feniks_architecture_20k_h100.slurm").read_text()
    submit = (ROOT / "scripts" / "submit_feniks_architecture_20k.sh").read_text()
    selector = (ROOT / "scripts" / "select_feniks_architecture_20k.py").read_text()

    assert "#SBATCH --array=0-5%6" in worker
    assert '--train-indices-file "$TRAIN_INDICES"' in worker
    assert '--validation-indices-file "$VALIDATION_INDICES"' in worker
    assert '--posterior-samples "$POSTERIOR_SAMPLES"' in worker
    assert "--min-median-ess-fraction 0.05" in worker
    assert "--max-fraction-pareto-k-gt-0p7 0.20" in worker
    assert "--n-train 18000 --n-validation 2000" in submit
    assert "truth_used_for_training_or_selection" in selector
