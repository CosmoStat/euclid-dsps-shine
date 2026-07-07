from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.amortized.train import (
    LossBatch,
    _pad_epoch_order_for_data_parallel,
    _resolve_data_parallel_training,
    _shard_loss_batch,
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
