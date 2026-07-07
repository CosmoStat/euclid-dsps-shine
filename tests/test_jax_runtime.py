from __future__ import annotations

import os

from euclid_dsps.jax_runtime import apply_jax_runtime_env, configure_jax_runtime


def test_runtime_auto_clears_forced_jax_platform(monkeypatch) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    apply_jax_runtime_env({"jax_platforms": "auto", "require_gpu": False})

    assert os.environ["EUCLID_DSPS_JAX_PLATFORMS"] == "auto"

    configure_jax_runtime()

    assert "JAX_PLATFORMS" not in os.environ


def test_runtime_cpu_overrides_forced_jax_platform(monkeypatch) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    apply_jax_runtime_env({"jax_platforms": "cpu", "require_gpu": False})

    assert os.environ["EUCLID_DSPS_JAX_PLATFORMS"] == "cpu"
    assert os.environ["JAX_PLATFORMS"] == "cpu"

    configure_jax_runtime()

    assert os.environ["JAX_PLATFORMS"] == "cpu"
