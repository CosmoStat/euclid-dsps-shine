"""JAX runtime configuration used before importing JAX-heavy modules."""

from __future__ import annotations

import os
from typing import Any


def apply_jax_runtime_env(runtime_config: dict[str, Any] | None) -> None:
    """Apply CLI/config runtime choices before importing JAX-heavy modules."""
    runtime = runtime_config or {}
    platforms = runtime.get("jax_platforms")
    if platforms:
        os.environ["EUCLID_DSPS_JAX_PLATFORMS"] = str(platforms)
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
    if "require_gpu" in runtime:
        os.environ.setdefault(
            "EUCLID_DSPS_REQUIRE_GPU",
            _bool_env(runtime["require_gpu"]),
        )
    if runtime.get("expected_gpu_name"):
        os.environ.setdefault(
            "EUCLID_DSPS_EXPECTED_GPU_NAME",
            str(runtime["expected_gpu_name"]),
        )
    if runtime.get("jax_compilation_cache_dir"):
        os.environ.setdefault(
            "EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR",
            str(runtime["jax_compilation_cache_dir"]),
        )
    if runtime.get("jax_persistent_cache_min_compile_time_secs") is not None:
        os.environ.setdefault(
            "EUCLID_DSPS_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
            str(runtime["jax_persistent_cache_min_compile_time_secs"]),
        )


def configure_jax_runtime() -> None:
    """Set conservative JAX defaults unless caller already configured them."""
    requested = os.environ.get("EUCLID_DSPS_JAX_PLATFORMS")
    if requested and requested.lower() == "auto":
        os.environ.pop("JAX_PLATFORMS", None)
    elif requested:
        os.environ.setdefault("JAX_PLATFORMS", requested)
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        os.environ.get("EUCLID_DSPS_XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
    )
    if _truthy(os.environ.get("EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD", "1")):
        import jax._src.xla_bridge as xla_bridge

        xla_bridge.discover_pjrt_plugins = lambda: None
    _configure_persistent_cache()
    if _truthy(os.environ.get("EUCLID_DSPS_REQUIRE_GPU", "0")):
        require_jax_gpu(os.environ.get("EUCLID_DSPS_EXPECTED_GPU_NAME"))


def _configure_persistent_cache() -> None:
    cache_dir = os.environ.get("EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        return
    from pathlib import Path

    path = Path(cache_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    import jax

    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    min_compile = os.environ.get(
        "EUCLID_DSPS_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
    )
    if min_compile is not None:
        jax.config.update(
            "jax_persistent_cache_min_compile_time_secs", float(min_compile)
        )


def require_jax_gpu(expected_name: str | None = None) -> list[str]:
    """Raise if JAX did not expose an NVIDIA/CUDA-capable GPU device."""
    import jax

    devices = jax.devices()
    gpu_devices = [
        device
        for device in devices
        if str(getattr(device, "platform", "")).lower() in {"cuda", "gpu"}
    ]
    if not gpu_devices:
        details = ", ".join(_device_label(device) for device in devices) or "none"
        raise RuntimeError(
            "JAX did not expose a CUDA/GPU device. "
            f"Visible JAX devices: {details}. "
            f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')!r}. "
            "Check that the shine environment has a CUDA-enabled jaxlib, "
            "EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0, and WSL nvidia-smi works."
        )
    labels = [_device_label(device) for device in gpu_devices]
    if expected_name:
        expected = expected_name.lower()
        if not any(expected in label.lower() for label in labels):
            raise RuntimeError(
                f"JAX GPU device does not match expected name {expected_name!r}. "
                f"Visible GPU devices: {', '.join(labels)}."
            )
    return labels


def _device_label(device: Any) -> str:
    platform = getattr(device, "platform", "unknown")
    kind = getattr(device, "device_kind", "")
    return f"{platform}:{kind}:{device}"


def _truthy(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _bool_env(value: Any) -> str:
    return "1" if bool(value) else "0"
