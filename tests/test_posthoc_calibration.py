from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.amortized.flows import RealNVPPrior
from euclid_dsps.amortized.posthoc_calibration import (
    PosteriorBank,
    _prior_loss_and_grad,
    build_defensive_mixture_bank,
    defensive_mixture_log_prob,
    importance_weight_bank,
    posterior_sample_paths,
    run_importance_correction,
    weighted_redshift_metrics,
)

eqx, optax = require_amortized_dependencies()


def _proposal_frame(n_samples: int = 32) -> pd.DataFrame:
    rows = []
    for row_index, center in ((10, 0.5), (20, 1.0)):
        values = np.linspace(center - 0.2, center + 0.2, n_samples)
        for sample_id, value in enumerate(values):
            rows.append(
                {
                    "object_id": 1000 + row_index,
                    "row_index": row_index,
                    "sample_id": sample_id,
                    "z_obs": value,
                    "nuisance": value - center,
                    "logq": -0.5 * ((value - center) / 0.1) ** 2,
                    "logprior": 0.0,
                    "loglike": -0.5 * ((value - (center + 0.1)) / 0.04) ** 2,
                }
            )
    return pd.DataFrame(rows)


def test_importance_weights_are_per_object_and_resampling_is_joint() -> None:
    frame = _proposal_frame()
    bank = PosteriorBank(
        frame=frame,
        identity_column="row_index",
        parameter_names=("z_obs", "nuisance"),
        source_files=(),
    )

    weighted, resampled, diagnostics = importance_weight_bank(
        bank, resample_count=64, seed=7
    )

    sums = weighted.groupby("row_index")[["raw_weight", "psis_weight"]].sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1.0e-12)
    assert len(resampled) == 2 * 64
    assert len(diagnostics) == 2
    np.testing.assert_allclose(
        resampled["nuisance"].to_numpy(),
        resampled["z_obs"].to_numpy()
        - resampled["row_index"].map({10: 0.5, 20: 1.0}).to_numpy(),
    )


def test_defensive_mixture_uses_realized_allocation_in_exact_density() -> None:
    base_frame = _proposal_frame(n_samples=8)
    tail_frame = _proposal_frame(n_samples=8)
    tail_frame["z_obs"] += 0.05
    base_bank = PosteriorBank(
        frame=base_frame,
        identity_column="row_index",
        parameter_names=("z_obs", "nuisance"),
        source_files=(),
    )
    tail_bank = PosteriorBank(
        frame=tail_frame,
        identity_column="row_index",
        parameter_names=("z_obs", "nuisance"),
        source_files=(),
    )
    base_q_base = np.linspace(-2.0, -1.0, len(base_frame))
    tail_q_base = base_q_base - 0.5
    base_q_tail = np.linspace(-3.0, -2.0, len(tail_frame))
    tail_q_tail = base_q_tail + 0.25

    bank, contract = build_defensive_mixture_bank(
        base_bank,
        tail_bank,
        base_logq_on_base=base_q_base,
        tail_logq_on_base=tail_q_base,
        base_logq_on_tail=base_q_tail,
        tail_logq_on_tail=tail_q_tail,
        base_target_logprior=np.full(len(base_frame), -4.0),
        tail_target_logprior=np.full(len(tail_frame), -5.0),
        requested_tail_fraction=0.26,
        seed=7,
    )

    assert contract["tail_draws_per_object"] == 2
    assert contract["base_draws_per_object"] == 6
    assert contract["realized_tail_fraction"] == 0.25
    counts = bank.frame.groupby(["row_index", "proposal_component"]).size()
    assert set(counts.xs("base", level="proposal_component")) == {6}
    assert set(counts.xs("tail", level="proposal_component")) == {2}
    expected = defensive_mixture_log_prob(
        bank.frame["logq_base"].to_numpy(),
        bank.frame["logq_tail"].to_numpy(),
        tail_fraction=0.25,
    )
    np.testing.assert_allclose(bank.frame["logq"], expected)
    assert set(
        bank.frame.loc[bank.frame["proposal_component"] == "base", "logprior"]
    ) == {-4.0}
    assert set(
        bank.frame.loc[bank.frame["proposal_component"] == "tail", "logprior"]
    ) == {-5.0}


def test_weighted_redshift_metrics_use_distributional_weights() -> None:
    frame = _proposal_frame()
    bank = PosteriorBank(
        frame=frame,
        identity_column="row_index",
        parameter_names=("z_obs", "nuisance"),
        source_files=(),
    )
    weighted, _resampled, _diagnostics = importance_weight_bank(bank)
    truth = pd.DataFrame(
        {
            "row_index": [10, 20],
            "object_id": [1010, 1020],
            "redshift_true": [0.6, 1.1],
        }
    )

    objects, summary = weighted_redshift_metrics(weighted, truth)

    assert len(objects) == 2
    assert summary["n_objects"] == 2
    assert summary["rmse"] < 0.03
    assert {"pit", "z_q16", "z_q84", "z_q025", "z_q975"} <= set(objects)


def test_importance_correction_writes_auditable_joint_outputs(tmp_path) -> None:
    posterior = tmp_path / "inference" / "posterior_samples"
    posterior.mkdir(parents=True)
    _proposal_frame().to_parquet(posterior / "batch_000001.parquet", index=False)
    truth = tmp_path / "truth.parquet"
    pd.DataFrame(
        {
            "row_index": [10, 20],
            "object_id": [1010, 1020],
            "redshift_true": [0.6, 1.1],
        }
    ).to_parquet(truth, index=False)
    out = tmp_path / "corrected"

    summary = run_importance_correction(
        posterior=posterior.parent,
        out_dir=out,
        truth_path=truth,
        resample_count=16,
    )

    assert summary["status"] == "complete"
    assert (out / "DONE").is_file()
    assert (out / "support_gate.json").is_file()
    assert (out / "importance_diagnostics.parquet").is_file()
    assert (out / "redshift_raw_weighted_objects.parquet").is_file()
    assert (out / "redshift_psis_weighted_objects.parquet").is_file()
    assert (out / "redshift_raw_weighted_summary.json").is_file()
    assert (out / "redshift_psis_weighted_summary.json").is_file()
    assert set(summary["redshift_metrics_by_weight"]) == {"raw", "psis"}
    assert summary["redshift_metrics"] == summary["redshift_metrics_by_weight"]["psis"]
    samples = pd.read_parquet(out / "resampled_samples" / "batch_000000.parquet")
    assert len(samples) == 32
    assert "source_sample_id" in samples
    assert posterior_sample_paths(posterior.parent) == (
        posterior / "batch_000001.parquet",
    )


def test_empirical_bayes_mstep_improves_weighted_prior_objective() -> None:
    prior = RealNVPPrior(
        jax.random.PRNGKey(1),
        latent_dim=2,
        n_layers=2,
        hidden_size=8,
        init="identity",
        init_scale=0.0,
    )
    x = jnp.asarray(
        [
            [[1.8, 1.9], [2.0, 2.1], [2.2, 2.0]],
            [[1.7, 2.2], [2.1, 1.8], [1.9, 2.0]],
        ],
        dtype=jnp.float32,
    )
    weights = jnp.full((2, 3), 1.0 / 3.0)
    trust_x = jnp.zeros((4, 2), dtype=jnp.float32)
    optimizer = optax.adam(1.0e-2)
    state = optimizer.init(eqx.filter(prior, eqx.is_inexact_array))

    first, grads = _prior_loss_and_grad(prior, x, weights, trust_x, 0.0)
    for _ in range(20):
        _loss, grads = _prior_loss_and_grad(prior, x, weights, trust_x, 0.0)
        updates, state = optimizer.update(
            grads, state, eqx.filter(prior, eqx.is_inexact_array)
        )
        prior = eqx.apply_updates(prior, updates)
    final, _ = _prior_loss_and_grad(prior, x, weights, trust_x, 0.0)

    assert float(final) < float(first)


def test_jean_zay_workflows_preserve_train_eval_and_distribution_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    importance = (root / "scripts/posthoc_importance_probe_h100.slurm").read_text()
    submit_importance = (
        root / "scripts/submit_posthoc_importance_probes.sh"
    ).read_text()
    assert "BUDGETS_EXPORT" in submit_importance
    assert 'BUDGETS_CSV="${BUDGETS_CSV//:/,}"' in importance
    empirical_bayes = (root / "scripts/posthoc_empirical_bayes_h100.slurm").read_text()

    assert "--no-posterior-predictive" in importance
    assert "importance_correct_posterior.py" in importance
    assert "TRAIN_INDICES" in empirical_bayes
    assert "EVAL_INDICES" in empirical_bayes
    assert "farmer_a24_n40000.parquet" in empirical_bayes
    assert "farmer_a24_full.parquet" in empirical_bayes
    assert "alternating_em_summary.json" in empirical_bayes
    assert "selected_candidate" in empirical_bayes
    assert "heldout_evidence_or_support_rejected_update" in empirical_bayes
    assert 'UPDATED_CHECKPOINT="$OUT/checkpoints/best.eqx"' in empirical_bayes
    assert "--fixed-feature-stats" in empirical_bayes
    assert "--freeze-prior" in empirical_bayes
    assert "--wake-every-encoder-epochs 1" in empirical_bayes
    assert "prior_frozen_exactly" in empirical_bayes
    assert "evaluate_feniks_mira.py" in empirical_bayes
    assert "evaluate_feniks_tarp.py" in empirical_bayes
    assert (
        "median"
        not in " ".join(
            line for line in empirical_bayes.splitlines() if "posterior" in line.lower()
        ).lower()
    )
