from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from euclid_dsps.amortized.train import (
    LossBatch,
    _pad_epoch_order_for_data_parallel,
    _replicate_tree,
    _resolve_data_parallel_training,
    _shard_loss_batch,
    _write_overlay_corner_like,
)
from euclid_dsps.cli import build_parser
from euclid_dsps.prior_learning.train import (
    _resolve_data_parallel_training as _resolve_prior_data_parallel_training,
)


def test_auto_data_parallel_falls_back_to_single_on_one_device() -> None:
    mode = _resolve_data_parallel_training(
        {"data_parallel": "auto"},
        jax_batch_size=8,
    )

    if mode["n_devices"] == 1:
        assert mode["effective"] == "single"
        assert mode["enabled"] is False
        assert mode["per_device_batch_size"] == 8


def test_forced_pmap_requires_multiple_devices() -> None:
    devices = _resolve_data_parallel_training(
        {"data_parallel": "single"},
        jax_batch_size=8,
    )["n_devices"]
    if devices > 1:
        pytest.skip("multi-device host can satisfy forced pmap")

    with pytest.raises(ValueError, match="requires at least two local JAX devices"):
        _resolve_data_parallel_training({"data_parallel": "pmap"}, jax_batch_size=8)


def test_pmap_epoch_padding_duplicates_without_dropping_rows() -> None:
    order = np.arange(10, dtype=np.int64)
    padded, n_padded = _pad_epoch_order_for_data_parallel(
        order,
        global_batch_size=8,
        enabled=True,
        rng=np.random.default_rng(1),
    )

    assert len(padded) == 16
    assert n_padded == 6
    assert set(order).issubset(set(padded.tolist()))


def test_shard_loss_batch_splits_leading_axis() -> None:
    batch = LossBatch(
        flux=jnp.ones((8, 3), dtype=jnp.float32),
        flux_err=2.0 * jnp.ones((8, 3), dtype=jnp.float32),
        mask=jnp.ones((8, 3), dtype=bool),
        features=jnp.ones((8, 6), dtype=jnp.float32),
        truth_theta=jnp.ones((8, 4), dtype=jnp.float32),
    )

    sharded = _shard_loss_batch(batch, 4)

    assert sharded.flux.shape == (4, 2, 3)
    assert sharded.flux_err.shape == (4, 2, 3)
    assert sharded.mask.shape == (4, 2, 3)
    assert sharded.features.shape == (4, 2, 6)
    assert sharded.truth_theta.shape == (4, 2, 4)


def test_replicate_tree_adds_device_axis_without_deprecated_jax_helper() -> None:
    devices = tuple(range(3))
    tree = {"weights": jnp.arange(6, dtype=jnp.float32).reshape(2, 3)}

    replicated = _replicate_tree(tree, devices)

    assert replicated["weights"].shape == (3, 2, 3)
    assert np.allclose(
        np.asarray(replicated["weights"][0]), np.asarray(tree["weights"])
    )


def test_prior_pmap_step_handles_static_realnvp_leaves_on_fake_cpu_devices() -> None:
    code = textwrap.dedent("""
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import optax

        from euclid_dsps.amortized.flows import RealNVPPrior
        from euclid_dsps.prior_learning.train import (
            _make_prior_pmap_train_step,
            _replicate_tree,
            _shard_x_batch,
        )

        devices = tuple(jax.local_devices())
        assert len(devices) == 3, devices
        prior = RealNVPPrior(
            jax.random.PRNGKey(0),
            latent_dim=4,
            n_layers=2,
            hidden_size=8,
            init="identity",
            init_scale=0.0,
        )
        optimizer = optax.adamw(1e-3)
        opt_state = optimizer.init(eqx.filter(prior, eqx.is_inexact_array))
        step = _make_prior_pmap_train_step(optimizer)
        prior_replicated = _replicate_tree(prior, devices)
        opt_state_replicated = _replicate_tree(opt_state, devices)
        batch = _shard_x_batch(jnp.ones((6, 4), dtype=jnp.float32), len(devices))
        (
            _, _, loss, mean_log_prob, base_penalty, grad_norm,
            loss_finite, grads_finite, update,
        ) = step(prior_replicated, opt_state_replicated, batch)
        assert loss.shape == (3,)
        assert mean_log_prob.shape == (3,)
        assert base_penalty.shape == (3,)
        assert grad_norm.shape == (3,)
        assert bool(loss_finite[0])
        assert bool(grads_finite[0])
        assert bool(update[0])
        """)
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=3"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_training_snapshot_corner_like_handles_diagonal_histograms(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        {
            "z_obs": np.linspace(0.1, 0.5, 32),
            "log10_stellar_mass": np.linspace(9.0, 10.0, 32),
        }
    )
    path = tmp_path / "corner.png"

    _write_overlay_corner_like(
        path,
        plt,
        columns=["z_obs", "log10_stellar_mass"],
        posterior=frame,
        prior=frame,
        truth=frame,
        max_rows=16,
    )

    assert path.exists()


def test_pmap_support_gate_skips_bad_selection_gradient_on_all_devices() -> None:
    code = textwrap.dedent("""
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import optax

        import euclid_dsps.amortized.train as train_module
        from euclid_dsps.amortized.elbo import AmortizedModel
        from euclid_dsps.amortized.flows import RealNVPPrior
        from euclid_dsps.amortized.posterior import ConditionalFlowEncoder
        from euclid_dsps.amortized.train import (
            JitLatentSpec,
            LossBatch,
            _make_pmap_train_step,
            _replicate_tree,
            _shard_loss_batch,
        )
        from euclid_dsps.calibration import GlobalSedScaleState

        devices = tuple(jax.local_devices())
        assert len(devices) == 2, devices
        encoder = ConditionalFlowEncoder(
            jax.random.PRNGKey(0), input_dim=6, latent_dim=4,
            hidden_sizes=(8,), activation="gelu", log_std_min=-6.0,
            log_std_max=2.0, initial_log_std=-1.0, family="realnvp",
            n_layers=2, hidden_size=8, output_space="latent_x",
        )
        model = AmortizedModel(
            encoder=encoder,
            prior=RealNVPPrior(
                jax.random.PRNGKey(1), latent_dim=4, n_layers=2,
                hidden_size=8, init="identity", init_scale=0.0,
            ),
            sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
        )
        batch = LossBatch(
            flux=jnp.ones((4, 2)), flux_err=0.1 * jnp.ones((4, 2)),
            mask=jnp.ones((4, 2), dtype=bool), features=jnp.ones((4, 6)),
            truth_theta=jnp.zeros((4, 0)),
        )
        spec = JitLatentSpec(
            names=("a", "b", "c", "d"), lower=jnp.zeros(4),
            upper=jnp.ones(4), raw_center=jnp.zeros(4), raw_scale=jnp.ones(4),
        )
        train_module.model_flux_from_x = lambda x, *_args, **_kwargs: x[..., :2]

        @jax.custom_jvp
        def bad(value):
            return -0.4 + 0.0 * value

        @bad.defjvp
        def bad_jvp(primals, tangents):
            (value,), (tangent,) = primals, tangents
            return bad(value), jnp.nan * tangent

        def fake_selection(candidate, *_args, **_kwargs):
            leaf = next(
                leaf for leaf in jax.tree_util.tree_leaves(candidate.prior)
                if eqx.is_inexact_array(leaf)
            )
            value = bad(jnp.sum(leaf))
            return value, {
                "selection/enabled": jnp.asarray(1.0),
                "selection/alpha": jnp.exp(value),
                "selection/log_alpha": value,
            }

        train_module._estimate_selection_log_alpha = fake_selection
        objective = {
            "mode": "reweighted_wake_sleep", "wake_active": True,
            "prior_train_jointly": True,
            "wake": {
                "n_particles": 4, "n_tempered_particles": 1,
                "base_temperature": 2.0, "train_encoder": False,
                "train_prior": True, "support_gate_enabled": True,
                "fail_median_ess_fraction": 1.1,
            },
            "selection_correction": {"enabled": True},
        }
        optimizer = optax.adam(1e-3)
        state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        step = _make_pmap_train_step(optimizer, gradient_clip_norm=5.0)
        result = step(
            _replicate_tree(model, devices), _replicate_tree(state, devices),
            _shard_loss_batch(batch, 2), spec, None, None, spec.names,
            jax.random.split(jax.random.PRNGKey(3), 2), 1, 1.0,
            {"type": "gaussian", "error_floor_frac": 0.0}, {}, objective,
            "prior_wake", False, False,
        )
        metrics, grads_finite, update_applied = result[3], result[6], result[7]
        assert jnp.all(grads_finite)
        assert jnp.all(~update_applied)
        assert jnp.all(metrics["selection/evaluated"] == 0.0)
    """)
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_train_parser_accepts_data_parallel_override() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "amortized-train-diffsky",
            "--data-parallel",
            "pmap",
        ]
    )

    assert args.data_parallel == "pmap"


def test_supervised_prior_parser_accepts_data_parallel_override() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "diffsky-train-supervised-prior",
            "--data-parallel",
            "auto",
        ]
    )

    assert args.data_parallel == "auto"


def test_prior_auto_data_parallel_falls_back_to_single_on_one_device() -> None:
    mode = _resolve_prior_data_parallel_training(
        {"data_parallel": "auto"},
        batch_size=8,
    )

    if mode["n_devices"] == 1:
        assert mode["effective"] == "single"
        assert mode["enabled"] is False
