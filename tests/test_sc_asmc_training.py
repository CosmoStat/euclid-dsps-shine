from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

HAS_EQUINOX = importlib.util.find_spec("equinox") is not None
pytestmark = pytest.mark.skipif(not HAS_EQUINOX, reason="equinox is not installed")


def test_ema_updates_arrays_and_preserves_static_fields() -> None:
    from euclid_dsps.amortized.encoder import ResidualPhotometryEncoder
    from euclid_dsps.amortized.sc_asmc_training import (
        tree_semantic_hash,
        update_ema_encoder,
    )

    old = ResidualPhotometryEncoder(
        jax.random.PRNGKey(1),
        input_dim=6,
        latent_dim=3,
        trunk_width=8,
        residual_blocks=1,
        representation_width=5,
        context_dim=4,
    )
    new = ResidualPhotometryEncoder(
        jax.random.PRNGKey(2),
        input_dim=6,
        latent_dim=3,
        trunk_width=8,
        residual_blocks=1,
        representation_width=5,
        context_dim=4,
    )
    ema = update_ema_encoder(old, new, decay=0.5)

    assert ema.input_dim == old.input_dim
    assert ema.latent_dim == old.latent_dim
    assert tree_semantic_hash(ema) not in {
        tree_semantic_hash(old),
        tree_semantic_hash(new),
    }
    expected = 0.5 * old.input_projection.weight + 0.5 * new.input_projection.weight
    assert jnp.allclose(ema.input_projection.weight, expected)


def test_ema_rejects_invalid_decay() -> None:
    from euclid_dsps.amortized.encoder import ResidualPhotometryEncoder
    from euclid_dsps.amortized.sc_asmc_training import update_ema_encoder

    encoder = ResidualPhotometryEncoder(
        jax.random.PRNGKey(1),
        input_dim=6,
        latent_dim=3,
        trunk_width=8,
        residual_blocks=1,
        representation_width=5,
        context_dim=4,
    )
    with pytest.raises(ValueError, match="EMA decay"):
        update_ema_encoder(encoder, encoder, decay=1.0)


def test_component_resume_is_bound_to_workflow_config(tmp_path) -> None:
    from euclid_dsps.amortized.features import FeatureStats, feature_stats_hash
    from euclid_dsps.amortized.latent import latent_spec_from_config, latent_spec_hash
    from euclid_dsps.amortized.posterior_bank import sha256_file
    from euclid_dsps.amortized.sc_asmc_config import sc_asmc_em_config_hash
    from euclid_dsps.amortized.sc_asmc_training import validate_component_checkpoint
    from euclid_dsps.config import load_config

    config = load_config("configs/experiments/feniks_sc_asmc_em_r25.yaml")
    stats = FeatureStats(
        flux_scale=jnp.ones(18),
        err_scale=jnp.ones(18),
        band_names=tuple(f"band_{index}" for index in range(18)),
        append_mask=True,
    )
    latent = latent_spec_from_config(config)
    runtime = SimpleNamespace(config=config, feature_stats=stats, latent_spec=latent)
    checkpoint = tmp_path / "q.eqx"
    checkpoint.write_bytes(b"checkpoint")
    digest = sha256_file(checkpoint)
    checkpoint.with_suffix(".eqx.json").write_text(
        json.dumps(
            {
                "sha256": digest,
                "workflow_config_hash": sc_asmc_em_config_hash(config),
                "latent_transform_hash": latent_spec_hash(latent),
                "feature_stats_hash": feature_stats_hash(stats),
                "truth_used": False,
            }
        ),
        encoding="utf-8",
    )

    validate_component_checkpoint(checkpoint, digest, runtime)
    changed = deepcopy(config)
    changed["amortized"]["sc_asmc_em"]["q_distillation"]["epochs"] = 4
    changed_runtime = SimpleNamespace(
        config=changed,
        feature_stats=stats,
        latent_spec=latent,
    )
    with pytest.raises(ValueError, match="workflow configuration mismatch"):
        validate_component_checkpoint(checkpoint, digest, changed_runtime)


def test_gaussian_sleep_toy_recovers_conditional_and_keeps_entropy() -> None:
    import equinox as eqx
    import optax

    from euclid_dsps.amortized.elbo import AmortizedModel
    from euclid_dsps.amortized.flows import StandardNormalPrior
    from euclid_dsps.amortized.posterior import (
        ConditionalFlowEncoder,
        posterior_entropy_diagnostics,
        posterior_log_prob,
    )
    from euclid_dsps.calibration import GlobalSedScaleState

    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(10),
        input_dim=2,
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
    latent_key, noise_key = jax.random.split(jax.random.PRNGKey(11))
    target = jax.random.normal(latent_key, (256, 2))
    features = target + 0.4 * jax.random.normal(noise_key, target.shape)
    optimizer = optax.adam(3.0e-3)
    state = optimizer.init(eqx.filter(model.encoder, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(candidate, optimizer_state):
        def loss_fn(candidate_encoder):
            updated = eqx.tree_at(
                lambda value: value.encoder, candidate, candidate_encoder
            )
            return -jnp.mean(posterior_log_prob(updated, features, target))

        loss, gradients = eqx.filter_value_and_grad(loss_fn)(candidate.encoder)
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, candidate.encoder
        )
        next_encoder = eqx.apply_updates(candidate.encoder, updates)
        return (
            eqx.tree_at(lambda value: value.encoder, candidate, next_encoder),
            optimizer_state,
            loss,
        )

    initial_nll = float(-jnp.mean(posterior_log_prob(model, features, target)))
    for _ in range(100):
        model, state, _loss = step(model, state)
    final_nll = float(-jnp.mean(posterior_log_prob(model, features, target)))
    entropy = posterior_entropy_diagnostics(
        model, features[:32], jax.random.PRNGKey(12), n_samples=8
    )
    _mean, log_std = model.encoder(features)

    assert final_nll < initial_nll - 0.5
    assert jnp.isfinite(entropy["posterior_full_entropy_mc"])
    assert entropy["posterior_full_entropy_mc"] > -5.0
    assert jnp.min(log_std) > -3.999
