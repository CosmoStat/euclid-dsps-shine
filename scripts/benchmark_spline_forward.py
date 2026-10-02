"""Benchmark the real spline forward with explicit alternate local SSP assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps import photometry_quadrature
from euclid_dsps.amortized.latent import LatentSpec, theta_to_x
from euclid_dsps.amortized.posterior_target import (
    PosteriorObservation,
    posterior_log_target,
)
from euclid_dsps.config import load_config
from euclid_dsps.filters import load_filters
from euclid_dsps.model import dynamic_model_args, load_context
from euclid_dsps.parameter_vectors import model_mags_from_theta_matrix_jax
from euclid_dsps.photometry import abmag_to_fnu_cgs_jax
from euclid_dsps.prior_learning.spline15d_schema import SPLINE15D_PARAMETER_NAMES
from scripts.benchmark_merged_photometry import baseline, time_function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssp", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--nuts-steps", type=int, default=0)
    parser.add_argument("--nuts-chains", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.nuts_steps < 0
        or args.nuts_chains < 1
        or args.out.exists()
    ):
        parser.error("positive repeats and fresh output required")
    jax.config.update("jax_enable_x64", True)
    config = load_config(
        "configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml"
    )
    bands = [band for band in config["bands"] if band["filter"]["kind"] == "ascii"]
    model_config = dict(
        config["model"],
        ssp_model="dense",
        photometry_integrator="merged_gauss4_v1",
        mdf_weight_precision="float64_v1",
        spline_precision="float64_v1",
    )
    context = load_context(
        str(args.ssp), load_filters(bands), n_sfh_bins=80, model_config=model_config
    )
    model_args = dynamic_model_args(context)
    theta = jnp.array([0.5, 9.3, -0.2, 0.3, -0.2] + [0.02] * 10, dtype=jnp.float64)
    spec = LatentSpec(
        SPLINE15D_PARAMETER_NAMES,
        jnp.array([0.01, 7.0, -2.0, 0.0, -2.0] + [-3.0] * 10),
        jnp.array([4.0, 12.0, 1.0, 3.0, 1.0] + [3.0] * 10),
        arithmetic_precision="float64_v1",
    )
    point = theta_to_x(theta, spec)
    center = abmag_to_fnu_cgs_jax(
        model_mags_from_theta_matrix_jax(context, model_args, theta, spec.names)
    )
    observation = PosteriorObservation(
        center[None, :] * 1.1,
        abs(center[None, :]) * 0.1,
        jnp.ones((1, len(bands)), bool),
    )
    model = SimpleNamespace(
        prior=SimpleNamespace(log_prob=lambda x: -0.5 * jnp.sum(x * x, axis=-1))
    )
    implementations = {
        "baseline": baseline,
        "current": photometry_quadrature.merged_integral_jax,
    }
    results, outputs = {}, {}
    try:
        for name, implementation in implementations.items():
            photometry_quadrature.merged_integral_jax = implementation

            def forward(point, arrays):
                return model_mags_from_theta_matrix_jax(
                    context, arrays, point, SPLINE15D_PARAMETER_NAMES
                )

            def objective(point, arrays):
                return jnp.sum(forward(point, arrays))

            for mode, function in (
                ("forward", forward),
                ("gradient", jax.value_and_grad(objective)),
            ):
                print(f"Benchmarking {name} {mode}", flush=True)
                outputs[name, mode], results[f"{name}_{mode}"] = time_function(
                    function, (theta, model_args), args.repeats
                )
                print(json.dumps(results[f"{name}_{mode}"]), flush=True)
            if args.nuts_steps:
                import blackjax

                def target(position, arrays, flux, error):
                    return posterior_log_target(
                        model,
                        position[None, None, :],
                        PosteriorObservation(flux, error, observation.mask),
                        spec,
                        context,
                        arrays,
                        spec.names,
                        {"type": "gaussian", "arithmetic_precision": "float64_v1"},
                    ).logtarget[0, 0]

                outputs[name, "target"], results[f"{name}_target"] = time_function(
                    jax.value_and_grad(target),
                    (point, model_args, observation.flux, observation.flux_err),
                    args.repeats,
                )
                print(json.dumps(results[f"{name}_target"]), flush=True)

                def run_nuts(x, arrays, flux, error, key):
                    def logdensity(position):
                        return target(position, arrays, flux, error)

                    algorithm = blackjax.nuts(
                        logdensity,
                        step_size=0.01,
                        inverse_mass_matrix=jnp.ones(15),
                        max_num_doublings=4,
                    )
                    state = algorithm.init(x)

                    def step(state, key):
                        state, info = algorithm.step(key, state)
                        return state, (
                            state.position,
                            info.num_integration_steps,
                            info.is_divergent,
                        )

                    return jax.lax.scan(
                        step, state, jax.random.split(key, args.nuts_steps)
                    )

                print(f"Benchmarking {name} NUTS", flush=True)
                starts = jnp.broadcast_to(point, (args.nuts_chains, len(point)))
                keys = jax.random.split(jax.random.PRNGKey(261001), args.nuts_chains)
                sample_result, timing = time_function(
                    jax.vmap(run_nuts, in_axes=(0, None, None, None, 0)),
                    (
                        starts,
                        model_args,
                        observation.flux,
                        observation.flux_err,
                        keys,
                    ),
                    min(args.repeats, 3),
                )
                timing["transitions"] = args.nuts_steps
                timing["chains"] = args.nuts_chains
                timing["integration_steps"] = np.asarray(sample_result[1][1]).tolist()
                timing["divergences"] = int(np.asarray(sample_result[1][2]).sum())
                results[f"{name}_nuts"] = timing
                print(json.dumps(timing), flush=True)
    finally:
        photometry_quadrature.merged_integral_jax = implementations["current"]
    parity = {}
    for mode in (
        ("forward", "gradient", "target")
        if args.nuts_steps
        else ("forward", "gradient")
    ):
        for i, (old, new) in enumerate(
            zip(
                jax.tree.leaves(outputs["baseline", mode]),
                jax.tree.leaves(outputs["current", mode]),
                strict=True,
            )
        ):
            np.testing.assert_allclose(new, old, rtol=1e-8, atol=1e-8)
            parity[f"{mode}_{i}_max_abs"] = float(
                np.max(np.abs(np.asarray(new) - np.asarray(old)))
            )
    report = dict(
        ssp=str(args.ssp.resolve()),
        model_config=model_config,
        bands=[b["name"] for b in bands],
        parameter_names=list(SPLINE15D_PARAMETER_NAMES),
        theta=np.asarray(theta).tolist(),
        devices=[str(d) for d in jax.devices()],
        jax_version=jax.__version__,
        timings=results,
        nuts_settings=dict(
            steps=args.nuts_steps,
            chains=args.nuts_chains,
            step_size=0.01,
            max_num_doublings=4,
            warmup=0,
            prior="standard Gaussian latent x",
            likelihood={"type": "gaussian", "arithmetic_precision": "float64_v1"},
        ),
        parity=parity,
        limitations=[
            "Alternate real dense SSP asset; production compressed basis absent.",
            "Four real Euclid filters; production has eighteen bands.",
            "Gradient of sum of magnitudes, not a production likelihood or learned prior.",
            "Single physical point; no MCMC convergence measurement.",
            "Optional NUTS probe uses Gaussian prior/noise, fixed step 0.01, depth4, no adaptation; performance only.",
        ],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
