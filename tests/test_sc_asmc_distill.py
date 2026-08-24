from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from euclid_dsps.amortized.sc_asmc_distill import (
    _pad_q_batch,
    _ratio_padded_bank_batches,
    _standard_smc_checkpoint_diagnostics,
    iter_bank_feature_batches,
)
from euclid_dsps.amortized.sc_asmc_mstep import _posterior_from_bank_shard
from tests.test_posterior_bank import _provenance, _shard


def test_q_distillation_padding_marks_duplicates_ineligible() -> None:
    shard = _shard(0, 3)
    posterior = _posterior_from_bank_shard(shard, np.arange(3))
    features = jnp.ones((3, 36))

    padded_features, padded = _pad_q_batch(features, posterior, 8)

    assert padded_features.shape == (8, 36)
    assert padded.particles.shape == (4, 8, 2)
    np.testing.assert_array_equal(np.asarray(padded.eligible), [1, 1, 1, 0, 0, 0, 0, 0])


def test_q_distillation_stream_padding_preserves_exact_three_to_one_ratio() -> None:
    shard = _shard(0, 2)
    posterior = _posterior_from_bank_shard(shard, np.arange(2))
    batches = [(jnp.full((2, 36), value), posterior) for value in (1.0, 2.0)]

    padded = list(_ratio_padded_bank_batches(iter(batches), ratio=3))

    assert len(padded) == 3
    assert [is_padding for _features, _posterior, is_padding in padded] == [
        False,
        False,
        True,
    ]
    np.testing.assert_array_equal(np.asarray(padded[-1][0]), np.asarray(batches[0][0]))


def test_q_checkpoint_hard_fraction_is_measured_before_extended_smc() -> None:
    result = SimpleNamespace(
        primary_indices=np.asarray([1, 2, 3]),
        primary=SimpleNamespace(number_of_stages=jnp.asarray([4, 6, 8])),
        fallback_indices=np.asarray([2, 3]),
        fallback=SimpleNamespace(number_of_stages=jnp.asarray([9, 11])),
        extended_indices=np.asarray([3]),
    )

    median_stages, hard_fraction = _standard_smc_checkpoint_diagnostics(
        result,
        n_objects=4,
    )

    assert median_stages == 9.0
    assert hard_fraction == 0.25


def test_full_bank_q_distillation_streams_every_resolved_object(tmp_path) -> None:
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.adaptive_smc_training import smc_q_distillation_loss
    from euclid_dsps.amortized.elbo import AmortizedModel
    from euclid_dsps.amortized.flows import StandardNormalPrior
    from euclid_dsps.amortized.posterior import ConditionalFlowEncoder
    from euclid_dsps.amortized.posterior_bank import (
        merge_posterior_bank_shards,
        write_posterior_bank_shard,
    )
    from euclid_dsps.calibration import GlobalSedScaleState

    write_posterior_bank_shard(tmp_path, 0, _shard(0, 3), _provenance())
    write_posterior_bank_shard(tmp_path, 1, _shard(3, 3), _provenance())
    manifest = merge_posterior_bank_shards(
        tmp_path,
        sorted((tmp_path / "shards").iterdir()),
        expected_row_indices=np.arange(6),
    )
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(21),
        input_dim=36,
        latent_dim=2,
        hidden_sizes=(8,),
        activation="gelu",
        log_std_min=-4.0,
        log_std_max=2.5,
        initial_log_std=0.0,
        family="realnvp",
        n_layers=2,
        hidden_size=16,
        scale_clamp=0.45,
        shift_clamp=3.0,
        init_scale=0.0,
        output_space="latent_x",
        context_encoder_type="residual_photometry",
        residual_trunk_width=16,
        residual_blocks=1,
        residual_representation_width=8,
        residual_context_dim=4,
        permutation="alternating_roll",
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=2),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    seen = 0
    for features, posterior in iter_bank_feature_batches(
        manifest,
        include_rows=set(range(6)),
        batch_size=2,
        shuffle_seed=3,
    ):
        loss, metrics = smc_q_distillation_loss(model, features, posterior)
        assert jnp.isfinite(loss)
        seen += int(metrics.eligible_count)

    assert seen == 6
