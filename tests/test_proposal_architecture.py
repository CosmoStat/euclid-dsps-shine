from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("equinox") is None,
    reason="equinox is not installed",
)


def _proposal(key, *, family="realnvp"):
    from euclid_dsps.amortized.proposal_architecture import (
        ContextualFlowProposal,
        ResidualContextEncoder,
    )

    context_key, flow_key = jax.random.split(key)
    context = ResidualContextEncoder(
        context_key,
        input_dim=7,
        context_dim=6,
        hidden_size=8,
        depth=2,
    )
    return ContextualFlowProposal(
        flow_key,
        context_encoder=context,
        context_dim=6,
        latent_dim=3,
        family=family,
        n_layers=2,
        hidden_size=8,
        n_bins=4,
        init_scale=0.01,
    )


@pytest.mark.parametrize("family", ["realnvp", "rq_spline"])
def test_contextual_flow_sample_density_agreement(family) -> None:
    from euclid_dsps.amortized.proposal_architecture import (
        contextual_log_prob,
        sample_contextual_proposal,
    )

    proposal = _proposal(jax.random.PRNGKey(1), family=family)
    observations = jax.random.normal(jax.random.PRNGKey(2), (4, 7))
    values, sampled_logq = sample_contextual_proposal(
        proposal, jax.random.PRNGKey(3), observations, 12
    )
    evaluated_logq = contextual_log_prob(proposal, observations, values)
    np.testing.assert_allclose(
        np.asarray(sampled_logq), np.asarray(evaluated_logq), rtol=2e-4, atol=2e-4
    )


def test_direct_context_retains_features_and_mask() -> None:
    from euclid_dsps.amortized.proposal_architecture import (
        make_band_token_observations,
        make_direct_observations,
    )

    features = jnp.arange(20, dtype=jnp.float32).reshape(2, 10)
    mask = jnp.asarray([[1, 1, 0, 1, 0], [1, 0, 1, 1, 1]], dtype=jnp.float32)
    direct = make_direct_observations(features, mask)
    tokens = make_band_token_observations(features, mask)
    assert direct.shape == (2, 15)
    assert tokens.shape == (2, 15)
    np.testing.assert_array_equal(np.asarray(direct[:, -5:]), np.asarray(mask))


def test_band_token_context_handles_missing_bands() -> None:
    from euclid_dsps.amortized.proposal_architecture import BandTokenContextEncoder

    encoder = BandTokenContextEncoder(
        jax.random.PRNGKey(4), n_bands=3, token_dim=8, context_dim=5
    )
    observations = jnp.asarray(
        [
            [1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 0.0, 0.0, 0.0],
        ]
    )
    result = encoder(observations)
    assert result.shape == (2, 5)
    assert bool(jnp.all(jnp.isfinite(result)))


def test_contextual_fit_uses_independent_validation_bank() -> None:
    from euclid_dsps.amortized.proposal_architecture import fit_contextual_proposal

    proposal = _proposal(jax.random.PRNGKey(5))
    observations = jax.random.normal(jax.random.PRNGKey(6), (6, 7))
    train = jax.random.normal(jax.random.PRNGKey(7), (16, 6, 3))
    validation = train + 1.5
    weights = jnp.full((16, 6), 1.0 / 16.0)
    result = fit_contextual_proposal(
        proposal,
        observations=observations,
        train_particles=train,
        train_weights=weights,
        validation_particles=validation,
        validation_weights=weights,
        train_indices=np.asarray([0, 1, 2]),
        validation_indices=np.asarray([3, 4, 5]),
        epochs=2,
        object_batch_size=2,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        seed=8,
        progress_label="test",
    )
    assert np.isfinite(result.initial_train_nll)
    assert np.isfinite(result.initial_validation_nll)
    assert result.initial_validation_nll != result.initial_train_nll


def test_phase1_selector_requires_every_seed_support(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts"
    import sys

    sys.path.insert(0, str(script))
    try:
        from select_popcosmos_proposal_architecture import _phase1_decision
    finally:
        sys.path.pop(0)

    def run(seed, ess, bad_k):
        return {
            "seed": seed,
            "validation_weighted_smc_b_nll": 12.0,
            "ordinary_is": {
                "validation": {
                    "support_status": (
                        "PASS" if ess >= 0.05 and bad_k <= 0.2 else "FAIL"
                    ),
                    "median_raw_ess_fraction": ess,
                    "fraction_pareto_k_gt_0p7": bad_k,
                }
            },
        }

    passing = [run(seed, 0.08, 0.1) for seed in (1, 2, 3)]
    failing = [run(1, 0.08, 0.1), run(2, 0.04, 0.1), run(3, 0.08, 0.1)]
    decision = _phase1_decision({"stable": passing, "unstable": failing})
    assert decision["selection_status"] == "PASS"
    assert decision["selected_candidate"] == "stable"
    unstable = next(
        item for item in decision["candidates"] if item["candidate"] == "unstable"
    )
    assert not unstable["eligible"]


def test_architecture_slurm_contract_is_self_supervised() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/submit_popcosmos_proposal_architecture.sh").read_text()
    worker = (root / "scripts/popcosmos_proposal_architecture_h100.slurm").read_text()
    evaluator = (
        root / "scripts/evaluate_popcosmos_proposal_architecture.py"
    ).read_text()
    assert "array_tasks=19" in launcher
    assert '--array="0-${array_max}%${ARRAY_CONCURRENCY}"' in launcher
    assert "train_particles=particles_a" in evaluator
    assert "validation_particles=particles_b" in evaluator
    assert "truth" not in worker.lower()


def test_report_pending_contract(tmp_path: Path, capsys) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/report_popcosmos_proposal_architecture.py"
    )
    namespace = {"__name__": "not_main"}
    exec(compile(script.read_text(), str(script), "exec"), namespace)
    import sys

    original = sys.argv
    try:
        sys.argv = [str(script), "--root", str(tmp_path)]
        namespace["main"]()
    finally:
        sys.argv = original
    output = capsys.readouterr().out
    assert "architecture_candidates_complete=0/19" in output
    assert "NEXT_ACTION=WAIT_FOR_PHASE01" in output
