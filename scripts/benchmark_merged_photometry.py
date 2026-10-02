"""Compare merged quadrature implementations on a real SSP spectrum."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.photometry_quadrature import merged_integral_jax


def baseline(wave, spectrum, filter_wave, transmission, z, *, order=4):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    shifted = wave * (1 + z)
    lo = jnp.maximum(shifted[0], filter_wave[0])
    hi = jnp.maximum(lo, jnp.minimum(shifted[-1], filter_wave[-1]))
    knots = jnp.sort(jnp.clip(jnp.concatenate((shifted, filter_wave)), lo, hi))
    widths = jnp.diff(knots)
    samples = knots[:-1, None] + widths[:, None] * (jnp.asarray(nodes) + 1) / 2
    lum = jnp.interp(samples, shifted, spectrum, left=0, right=0)
    trans = jnp.interp(samples, filter_wave, transmission, left=0, right=0)
    return jnp.sum(
        widths * jnp.sum(lum * trans / samples * jnp.asarray(weights), axis=-1) / 2
    )


def time_function(fn, arguments, repeats):
    started = time.perf_counter()
    executable = jax.jit(fn).lower(*arguments).compile()
    compile_seconds = time.perf_counter() - started
    values = jax.block_until_ready(executable(*arguments))
    if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(values)):
        raise ValueError("Non-finite benchmark values or gradients")
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(executable(*arguments))
        times.append(time.perf_counter() - started)
    return values, dict(
        compile_seconds=compile_seconds,
        median_ms=1000 * float(np.median(times)),
        p95_ms=1000 * float(np.quantile(times, 0.95)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssp", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.out.exists():
        parser.error("positive repeats and fresh output required")
    jax.config.update("jax_enable_x64", True)
    with h5py.File(args.ssp) as handle:
        wave = np.asarray(handle["ssp_wave"])
        spectrum = np.asarray(handle["ssp_flux"])[0, -10]
    fw = np.linspace(4500, 8500, 512)
    ft = np.exp(-0.5 * ((fw - 6500) / 800) ** 2)
    arguments = tuple(map(jnp.asarray, (wave, spectrum, fw, ft, 0.31)))
    results = {}
    outputs = {}
    for name, implementation in (
        ("baseline", baseline),
        ("current", merged_integral_jax),
    ):

        def function(w, s, f, t, z, implementation=implementation):
            return implementation(w, s, f, t, z, order=4)

        for mode, fn in (
            ("value", function),
            ("gradient", jax.value_and_grad(function, argnums=(1, 4))),
        ):
            outputs[name, mode], results[f"{name}_{mode}"] = time_function(
                fn, arguments, args.repeats
            )
    errors = {}
    for mode in ("value", "gradient"):
        for i, (old, new) in enumerate(
            zip(
                jax.tree.leaves(outputs["baseline", mode]),
                jax.tree.leaves(outputs["current", mode]),
                strict=True,
            )
        ):
            np.testing.assert_allclose(new, old, rtol=1e-9, atol=1e-12)
            errors[f"{mode}_{i}_max_abs"] = float(
                np.max(np.abs(np.asarray(new) - np.asarray(old)))
            )
    report = dict(
        ssp=str(args.ssp.resolve()),
        n_wave=len(wave),
        filter="synthetic Gaussian, 512 nodes",
        devices=[str(d) for d in jax.devices()],
        jax_version=jax.__version__,
        timings=results,
        parity=errors,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
