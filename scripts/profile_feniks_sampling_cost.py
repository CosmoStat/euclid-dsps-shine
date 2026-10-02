"""Measure canonical posterior gradient costs without running MCMC."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.amortized.posterior import sample_posterior
from euclid_dsps.amortized.posterior_target import (
    PosteriorObservation,
    posterior_log_target,
)
from euclid_dsps.config import load_config
from scripts.run_feniks_exact_posterior_benchmark import _load_runtime_rows


def measure(function, positions: jax.Array, repeats: int) -> dict:
    """Synchronize each call; compile and prime outside the measured repeats."""
    started = time.perf_counter()
    executable = jax.jit(jax.vmap(function)).lower(positions).compile()
    compile_seconds = time.perf_counter() - started
    result = jax.block_until_ready(executable(positions))
    if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(result)):
        raise ValueError("Non-finite values or gradients at profiling positions")
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(executable(positions))
        elapsed.append(time.perf_counter() - started)
    median = float(np.median(elapsed))
    return {
        "compile_seconds": compile_seconds,
        "batch_median_ms": median * 1000,
        "batch_p95_ms": float(np.quantile(elapsed, 0.95)) * 1000,
        "evaluations_per_second": len(positions) / median,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=261001)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.positions < 1 or args.repeats < 1 or args.row < 0:
        parser.error("positions/repeats must be positive and row nonnegative")
    if args.out.exists():
        parser.error("output already exists; choose a fresh output path")
    jax.config.update("jax_enable_x64", True)
    devices = jax.devices()
    if all(device.platform == "cpu" for device in devices) and not args.allow_cpu:
        parser.error("CPU-only JAX: use GPU JAX or explicitly pass --allow-cpu")
    config = load_config(args.config)
    config["catalog_path"] = str(args.dataset)
    started = time.perf_counter()
    runtime = _load_runtime_rows(args, config, np.array([args.row], dtype=np.int64))
    draws = sample_posterior(
        runtime.model,
        jax.random.PRNGKey(args.seed),
        runtime.batch.features,
        args.positions,
    )
    positions = jnp.asarray(draws.x[:, 0, :], dtype=jnp.float64)
    jax.block_until_ready(positions)
    load_seconds = time.perf_counter() - started
    observation = PosteriorObservation(
        runtime.batch.flux, runtime.batch.flux_err, runtime.batch.mask
    )

    def parts(x):
        return posterior_log_target(
            runtime.model,
            x[None, None, :],
            observation,
            runtime.latent_spec,
            runtime.context,
            runtime.model_args,
            runtime.latent_spec.names,
            runtime.likelihood,
            {"calibration": config.get("calibration", {}) or {}},
        )

    def target(x):
        return parts(x).logtarget[0, 0].astype(jnp.float64)

    def likelihood(x):
        return parts(x).loglike[0, 0].astype(jnp.float64)

    def prior(x):
        return parts(x).logprior[0, 0].astype(jnp.float64)

    def prior_network(x):
        return runtime.model.prior.log_prob(x[None, :])[0].astype(jnp.float64)

    measurements = {}
    for name, function in (
        ("target_value", target),
        ("target_value_and_gradient", jax.value_and_grad(target)),
        ("likelihood_value_and_gradient", jax.value_and_grad(likelihood)),
        ("guarded_prior_value_and_gradient", jax.value_and_grad(prior)),
        ("prior_network_value_and_gradient", jax.value_and_grad(prior_network)),
    ):
        print(f"Profiling {name}, positions={args.positions}", flush=True)
        measurements[name] = measure(function, positions, args.repeats)
        print(json.dumps(measurements[name]), flush=True)
    report = {
        "config": str(args.config.resolve()),
        "dataset": str(args.dataset.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "feature_stats": str(args.feature_stats.resolve()),
        "row": args.row,
        "seed": args.seed,
        "parameter_names": list(runtime.latent_spec.names),
        "positions": args.positions,
        "repeats": args.repeats,
        "devices": [str(device) for device in devices],
        "jax_version": jax.__version__,
        "input_dtype": str(positions.dtype),
        "model_config": config.get("model", {}),
        "likelihood_config": runtime.likelihood,
        "load_and_initialize_seconds": load_seconds,
        "measurements": measurements,
        "limitations": [
            "Encoder positions for one object, not a representative NUTS trajectory.",
            "Component timings are independently compiled and are not additive.",
            "Guarded canonical prior includes decoder flux-finiteness checks; network-only prior does not.",
            "Input float64 does not promote checkpoint or decoder internal arrays.",
            "No warmup, NUTS control flow, convergence or ESS measured here.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
