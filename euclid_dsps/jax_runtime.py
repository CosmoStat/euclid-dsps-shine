"""JAX runtime configuration used before importing JAX-heavy modules."""

from __future__ import annotations

import os
from typing import Any


def apply_jax_runtime_env(runtime_config: dict[str, Any] | None) -> None:
    """Apply CLI/config runtime choices before importing JAX-heavy modules."""
    runtime = runtime_config or {}
    platforms = runtime.get("jax_platforms")
    if platforms:
        os.environ.setdefault("EUCLID_DSPS_JAX_PLATFORMS", str(platforms))
    if "disable_jax_plugin_autoload" in runtime:
        os.environ.setdefault(
            "EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD",
            _bool_env(runtime["disable_jax_plugin_autoload"]),
        )
    if "xla_python_client_preallocate" in runtime:
        os.environ.setdefault(
            "EUCLID_DSPS_XLA_PYTHON_CLIENT_PREALLOCATE",
            _bool_env(runtime["xla_python_client_preallocate"]),
        )


def configure_jax_runtime() -> None:
    """Set conservative JAX defaults unless caller already configured them.
    """
    os.environ.setdefault(
        "JAX_PLATFORMS", os.environ.get("EUCLID_DSPS_JAX_PLATFORMS", "cpu")
    )
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        os.environ.get("EUCLID_DSPS_XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
    )
    if _truthy(os.environ.get("EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD", "1")):
        import jax._src.xla_bridge as xla_bridge

        xla_bridge.discover_pjrt_plugins = lambda: None


def _truthy(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _bool_env(value: Any) -> str:
    return "1" if bool(value) else "0"
