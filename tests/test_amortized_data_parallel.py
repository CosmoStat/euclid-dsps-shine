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
    )

    sharded = _shard_loss_batch(batch, 4)

    assert sharded.flux.shape == (4, 2, 3)
    assert sharded.flux_err.shape == (4, 2, 3)
    assert sharded.mask.shape == (4, 2, 3)
    assert sharded.features.shape == (4, 2, 6)


def test_replicate_tree_adds_device_axis_without_deprecated_jax_helper() -> None:
    devices = tuple(range(3))
    tree = {"weights": jnp.arange(6, dtype=jnp.float32).reshape(2, 3)}

    replicated = _replicate_tree(tree, devices)

    assert replicated["weights"].shape == (3, 2, 3)
    assert np.allclose(np.asarray(replicated["weights"][0]), np.asarray(tree["weights"]))


def test_prior_pmap_step_handles_static_realnvp_leaves_on_fake_cpu_devices() -> None:
    code = textwrap.dedent(
        """
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
        _, _, loss, mean_log_prob, grad_norm, loss_finite, grads_finite, update = step(
            prior_replicated,
            opt_state_replicated,
            batch,
        )
        assert loss.shape == (3,)
        assert mean_log_prob.shape == (3,)
        assert grad_norm.shape == (3,)
        assert bool(loss_finite[0])
        assert bool(grads_finite[0])
        assert bool(update[0])
        """
    )
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
