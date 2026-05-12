"""Check that JAX in the active environment really uses the NVIDIA GPU.

Run from the repository root:

    python scripts/check_jax_gpu.py --require-nvidia --hold-seconds 10

The script prints ``nvidia-smi`` visibility, JAX devices, selected backend, and
runs a real JAX matrix multiplication while GPU memory is still allocated.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platforms",
        default="cuda",
        help="Value for EUCLID_DSPS_JAX_PLATFORMS/JAX_PLATFORMS.",
    )
    parser.add_argument(
        "--require-nvidia",
        action="store_true",
        help="Fail if JAX does not expose a GPU whose device_kind contains NVIDIA.",
    )
    parser.add_argument(
        "--expected-name",
        default="NVIDIA",
        help="Expected substring in JAX GPU device_kind when --require-nvidia is set.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=4096,
        help="Square matrix size for the JAX matmul smoke test.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=5,
        help="Keep the result buffer alive so nvidia-smi/Task Manager can observe it.",
    )
    args = parser.parse_args()

    _print_header("Python")
    print(f"executable: {sys.executable}")
    print(f"cwd:        {Path.cwd()}")
    print(f"pid:        {os.getpid()}")

    _print_header("nvidia-smi before JAX")
    _run(["nvidia-smi"])
    gpu_query_before = _nvidia_memory_query()

    _configure_env(args)

    _print_header("Runtime env")
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "JAX_PLATFORMS",
        "EUCLID_DSPS_JAX_PLATFORMS",
        "EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "EUCLID_DSPS_XLA_PYTHON_CLIENT_PREALLOCATE",
        "EUCLID_DSPS_REQUIRE_GPU",
        "EUCLID_DSPS_EXPECTED_GPU_NAME",
    ):
        print(f"{key}={os.environ.get(key, '')}")

    from euclid_dsps.jax_runtime import configure_jax_runtime, require_jax_gpu

    configure_jax_runtime()

    import jax
    import jax.numpy as jnp

    _print_header("JAX devices")
    print(f"jax version:      {jax.__version__}")
    print(f"default backend:  {jax.default_backend()}")
    devices = jax.devices()
    for index, device in enumerate(devices):
        print(
            f"[{index}] platform={getattr(device, 'platform', '')} "
            f"kind={getattr(device, 'device_kind', '')} repr={device}"
        )

    if args.require_nvidia:
        labels = require_jax_gpu(args.expected_name)
        print(f"required GPU OK: {labels}")

    gpu_devices = [
        device
        for device in devices
        if str(getattr(device, "platform", "")).lower() in {"cuda", "gpu"}
    ]
    device = gpu_devices[0] if gpu_devices else devices[0]

    _print_header("JAX matmul smoke")
    print(f"selected device: {device}")
    print(f"matrix size:     {args.size}x{args.size}")
    x = jax.device_put(jnp.ones((args.size, args.size), dtype=jnp.float32), device)
    start = time.perf_counter()
    y = (x @ x).block_until_ready()
    elapsed = time.perf_counter() - start
    checksum = float(jnp.asarray(y[0, 0]).block_until_ready())
    print(f"elapsed_s:       {elapsed:.3f}")
    print(f"checksum:        {checksum:.6e}")

    _print_header("nvidia-smi after JAX allocation")
    gpu_query_after = _nvidia_memory_query()
    if gpu_query_before and gpu_query_after:
        print("memory before:")
        print(gpu_query_before)
        print("memory after:")
        print(gpu_query_after)
    _run(["nvidia-smi"])
    compute_apps = _nvidia_compute_apps_query()
    if compute_apps:
        print("compute apps:")
        print(compute_apps)
    if args.require_nvidia:
        _assert_current_pid_on_nvidia(compute_apps)

    if args.hold_seconds > 0:
        _print_header("hold")
        print(
            f"Holding result for {args.hold_seconds}s. "
            "Run nvidia-smi in another terminal if needed."
        )
        time.sleep(args.hold_seconds)


def _configure_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("EUCLID_DSPS_JAX_PLATFORMS", args.platforms)
    os.environ.setdefault("JAX_PLATFORMS", args.platforms)
    plugin_autoload_default = "0" if "cuda" in args.platforms.lower() else "1"
    os.environ.setdefault(
        "EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD", plugin_autoload_default
    )
    os.environ.setdefault("EUCLID_DSPS_XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.require_nvidia:
        os.environ.setdefault("EUCLID_DSPS_REQUIRE_GPU", "1")
        os.environ.setdefault("EUCLID_DSPS_EXPECTED_GPU_NAME", args.expected_name)


def _nvidia_memory_query() -> str:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        print_output=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _nvidia_compute_apps_query() -> str:
    result = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        print_output=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _assert_current_pid_on_nvidia(compute_apps: str) -> None:
    pid = str(os.getpid())
    if pid not in compute_apps:
        raise SystemExit(
            "JAX completed, but current Python PID was not visible in "
            "`nvidia-smi --query-compute-apps`. This means the run is not proven "
            "to be using the NVIDIA compute device. Re-run with "
            "`JAX_PLATFORMS=cuda`, `EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0`, "
            "and check `nvidia-smi` from the same WSL terminal."
        )


def _run(command: list[str], print_output: bool = True) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        print(f"{command[0]} not found")
        return subprocess.CompletedProcess(command, 127, "", f"{command[0]} not found")
    if print_output:
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip())
        if result.returncode:
            print(f"exit_code={result.returncode}")
    return result


def _print_header(title: str) -> None:
    print(f"\n== {title} ==")


if __name__ == "__main__":
    main()
