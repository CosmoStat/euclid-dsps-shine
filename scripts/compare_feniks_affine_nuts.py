"""Bounded raw/affine/flow-coordinate NUTS probes of the canonical posterior."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.amortized.exact_posterior import (
    _rank_normalize,
    autocorrelation_ess,
    split_rhat,
    tail_indicator_ess,
)
from euclid_dsps.amortized.posterior_target import (
    PosteriorObservation,
    posterior_log_target,
)
from scripts.benchmark_feniks_production_local import digest, load_inputs


def affine_from_bank(points, weights, shrinkage=0.05):
    points, weights = np.asarray(points, float), np.asarray(weights, float)
    if points.ndim != 2 or weights.shape != (len(points),) or not 0 < shrinkage <= 1:
        raise ValueError("Invalid points, weights or shrinkage")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("Nonfinite points or invalid weights")
    weights = weights / weights.sum()
    mean = weights @ points
    centered = points - mean
    covariance = (centered * weights[:, None]).T @ centered
    # The bank may have low importance ESS. Use proposal diagonal shrinkage,
    # not an unchecked nearly singular weighted covariance.
    diagonal = np.maximum(np.var(points, axis=0), 1e-6)
    covariance = (1 - shrinkage) * covariance + shrinkage * np.diag(diagonal)
    covariance += np.eye(points.shape[1]) * 1e-8 * max(float(np.max(diagonal)), 1.0)
    factor = np.linalg.cholesky(covariance)
    return mean, factor, float(1 / np.sum(weights**2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=48)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=261002)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("raw_dense", "affine_dense", "flow_dense"),
        default=["raw_dense", "affine_dense"],
    )
    args = parser.parse_args()
    if "flow_dense" in args.arms and args.arms != ["flow_dense"]:
        parser.error("flow_dense must run alone to isolate its compiled graph")
    if (
        args.out.exists()
        or args.warmup < 20
        or args.samples < 4
        or not 2 <= args.chains <= 8
        or not 1 <= args.depth <= 6
    ):
        parser.error(
            "Fresh output, warmup>=20, samples>=4, chains=2..8, depth=1..6 required"
        )
    jax.config.update("jax_enable_x64", True)
    (
        config_path,
        config,
        checkpoint,
        model,
        spec,
        context,
        arrays,
        _,
        flux,
        error,
        mask,
        provenance,
    ) = load_inputs(1)
    directory = Path(provenance[0]["directory"])
    starts_x = jnp.asarray(np.load(directory / "starts_B.npy")[: args.chains])
    with np.load(directory / "bank_0.npz") as bank:
        mean, factor, bank_ess = affine_from_bank(bank["x"], bank["weight"])
    transforms = {
        "raw_dense": (jnp.zeros(15), jnp.eye(15)),
        "affine_dense": (jnp.asarray(mean), jnp.asarray(factor)),
    }
    flow_mode = args.arms == ["flow_dense"]
    flow_provenance = {}
    if flow_mode:
        from euclid_dsps.amortized.features import (
            make_encoder_features,
            read_feature_stats,
        )
        from euclid_dsps.amortized.local_transport_precision import (
            DiagnosticTransport64,
        )
        from euclid_dsps.amortized.local_vi_diagnostic import initialize

        stats_path = Path("outputs/local_sampling_inputs/feature_stats.json")
        features = make_encoder_features(
            flux, error, read_feature_stats(stats_path), mask
        )
        parameters, flow_context = initialize(model, features)
        transport = DiagnosticTransport64(model.encoder)
        flow_context = jax.tree.map(lambda a: a.astype(jnp.float64), flow_context)
        base_mean = parameters.mean[0].astype(jnp.float64)
        base_logstd = jnp.clip(
            parameters.log_std[0],
            model.encoder.base.log_std_min,
            model.encoder.base.log_std_max,
        ).astype(jnp.float64)
        transforms = {"flow_dense": (jnp.zeros(15), jnp.eye(15))}
        flow_provenance = {"feature_stats_sha256": digest(stats_path)}

    def physical(q, mean, factor):
        if flow_mode:
            x, logdet = transport.forward(
                (base_mean + jnp.exp(base_logstd) * q)[None, :], flow_context
            )
            return x[0], logdet[0] + base_logstd.sum()
        return mean + factor @ q, jnp.log(jnp.diag(factor)).sum()

    def coordinates(x, mean, factor):
        if flow_mode:
            base, _ = transport.inverse(x[None, :], flow_context)
            return (base[0] - base_mean) * jnp.exp(-base_logstd)
        return jnp.linalg.solve(factor, x - mean)

    observation = PosteriorObservation(flux, error, mask)

    def raw_target(x, arrays):
        return posterior_log_target(
            model,
            x[None, None, :],
            observation,
            spec,
            context,
            arrays,
            spec.names,
            config["amortized"]["likelihood"],
            {"calibration": config.get("calibration", {})},
        ).logtarget[0, 0]

    def target(q, arrays, mean, factor):
        x, logdet = physical(q, mean, factor)
        return raw_target(x, arrays) + logdet

    # Validate the transformed density and chain rule before launching MCMC.
    mean_j, factor_j = transforms["flow_dense" if flow_mode else "affine_dense"]
    q = coordinates(starts_x[0], mean_j, factor_j)
    recovered, logdet = physical(q, mean_j, factor_j)
    np.testing.assert_allclose(recovered, starts_x[0], rtol=1e-9, atol=1e-9)
    raw_value, raw_gradient = jax.jit(jax.value_and_grad(raw_target))(
        starts_x[0], arrays
    )
    value, gradient = jax.jit(jax.value_and_grad(target))(q, arrays, mean_j, factor_j)
    np.testing.assert_allclose(value, raw_value + logdet, rtol=1e-9, atol=1e-7)
    jacobian = jax.jacfwd(lambda qq: physical(qq, mean_j, factor_j)[0])(q)
    np.testing.assert_allclose(
        logdet, jnp.linalg.slogdet(jacobian)[1], rtol=1e-9, atol=1e-9
    )
    determinant_gradient = jax.grad(lambda qq: physical(qq, mean_j, factor_j)[1])(q)
    np.testing.assert_allclose(
        gradient, jacobian.T @ raw_gradient + determinant_gradient, rtol=1e-8, atol=1e-7
    )

    def adapt(position, key, arrays, mean, factor):
        def density(q):
            return target(q, arrays, mean, factor)

        adaptation = blackjax.window_adaptation(
            blackjax.nuts,
            density,
            is_mass_matrix_diagonal=False,
            initial_step_size=0.01,
            target_acceptance_rate=0.9,
            max_num_doublings=args.depth,
            adaptation_info_fn=lambda state, info, adaptation_state: (
                info.num_integration_steps,
                info.is_divergent,
                info.acceptance_rate,
            ),
        )
        return adaptation.run(key, position, num_steps=args.warmup)

    def sample(state, parameters, key, arrays, mean, factor):
        def density(q):
            return target(q, arrays, mean, factor)

        algorithm = blackjax.nuts(
            density,
            step_size=parameters["step_size"],
            inverse_mass_matrix=parameters["inverse_mass_matrix"],
            max_num_doublings=args.depth,
        )

        def step(state, key):
            state, info = algorithm.step(key, state)
            return state, (
                physical(state.position, mean, factor)[0],
                info.num_integration_steps,
                info.is_divergent,
                info.acceptance_rate,
            )

        return jax.lax.scan(step, state, jax.random.split(key, args.samples))

    warm_keys = jax.random.split(jax.random.PRNGKey(args.seed), args.chains)
    sample_keys = jax.random.split(jax.random.PRNGKey(args.seed + 1), args.chains)
    adapt_fn = jax.jit(jax.vmap(adapt, in_axes=(0, 0, None, None, None)))
    sample_fn = jax.jit(jax.vmap(sample, in_axes=(0, 0, 0, None, None, None)))
    args.out.mkdir(parents=True)
    report = {
        "status": "TARGET_VERIFIED_SAMPLING_PENDING",
        "checkpoint_sha256": digest(checkpoint),
        "config_sha256": digest(config_path),
        "bank_sha256": digest(directory / "bank_0.npz"),
        "proposal_bank_importance_ess": bank_ess,
        "seed": args.seed,
        "chains": args.chains,
        "warmup": args.warmup,
        "samples_per_chain": args.samples,
        "depth": args.depth,
        "target_accept": 0.9,
        "devices": [d.device_kind for d in jax.devices()],
        "jax_version": jax.__version__,
        "blackjax_version": blackjax.__version__,
        "source_sha256": {
            path: digest(Path(path))
            for path in (
                "scripts/compare_feniks_affine_nuts.py",
                "euclid_dsps/model.py",
                "euclid_dsps/photometry_quadrature.py",
                "euclid_dsps/amortized/local_transport_precision.py",
            )
        },
        "target_value_and_gradient_transform_check": "PASS",
        "results": {},
        "requested_arms": args.arms,
        "transport_provenance": flow_provenance,
        "input_provenance": provenance,
        "limitations": [
            "Short diagnostic probe, not convergence or publication evidence.",
            "Same target and dense adaptation for both arms; no guarantee linear whitening helps.",
            "Affine covariance from a low-ESS proposal bank, stabilized by 5% proposal diagonal.",
            "Separate-process arms are not paired hardware timings; repeat before latency claims.",
            "Flow coordinates use float64 transport and exact Jacobian; the physical target is unchanged.",
        ],
    }
    np.savez_compressed(args.out / "affine.npz", mean=mean, factor=factor)
    (args.out / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    adapt_exe = sample_exe = None
    for label, (mean, factor) in transforms.items():
        if label not in args.arms:
            continue
        positions = jax.vmap(coordinates, in_axes=(0, None, None))(
            starts_x, mean, factor
        )
        recovered, _ = jax.vmap(physical, in_axes=(0, None, None))(
            positions, mean, factor
        )
        np.testing.assert_allclose(recovered, starts_x, rtol=1e-9, atol=1e-9)
        arguments = (positions, warm_keys, arrays, mean, factor)
        print(f"{label}: warmup", flush=True)
        start = time.perf_counter()
        if adapt_exe is None:
            adapt_exe = adapt_fn.lower(*arguments).compile()
            report["warmup_compile_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        adapted, warm_info = jax.block_until_ready(adapt_exe(*arguments))
        warm_seconds = time.perf_counter() - start
        arguments = (
            adapted.state,
            adapted.parameters,
            sample_keys,
            arrays,
            mean,
            factor,
        )
        print(f"{label}: sampling", flush=True)
        start = time.perf_counter()
        if sample_exe is None:
            sample_exe = sample_fn.lower(*arguments).compile()
            report["sampling_compile_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        final, info = jax.block_until_ready(sample_exe(*arguments))
        sample_seconds = time.perf_counter() - start
        draws, steps, divergences, acceptance = map(np.asarray, info)
        if (
            not np.isfinite(draws).all()
            or not np.isfinite(np.asarray(final.logdensity)).all()
        ):
            raise ValueError(f"Nonfinite chain state: {label}")
        diagnostics = [
            {
                "parameter": name,
                "rhat": split_rhat(draws[:, :, i]),
                "bulk_ess": autocorrelation_ess(_rank_normalize(draws[:, :, i])),
                "tail_ess": tail_indicator_ess(draws[:, :, i]),
            }
            for i, name in enumerate(spec.names)
        ]
        minimum_ess = min(d["bulk_ess"] for d in diagnostics)
        result = {
            "warmup_seconds": warm_seconds,
            "sampling_seconds": sample_seconds,
            "total_execution_seconds": warm_seconds + sample_seconds,
            "warmup_integrations": int(np.asarray(warm_info[0]).sum()),
            "sampling_integrations": int(steps.sum()),
            "mean_integration_steps": float(steps.mean()),
            "depth_saturation_fraction": float(np.mean(steps == 2**args.depth - 1)),
            "warmup_divergences": int(np.asarray(warm_info[1]).sum()),
            "sampling_divergences": int(divergences.sum()),
            "acceptance_mean": float(acceptance.mean()),
            "step_sizes": np.asarray(adapted.parameters["step_size"]).tolist(),
            "min_bulk_ess": minimum_ess,
            "max_rhat": max(d["rhat"] for d in diagnostics),
            "exploratory_min_ess_per_second": minimum_ess
            / (warm_seconds + sample_seconds),
            "diagnostics": diagnostics,
        }
        np.savez_compressed(
            args.out / f"{label}_draws.npz",
            x=draws,
            steps=steps,
            divergences=divergences,
            acceptance=acceptance,
        )
        report["results"][label] = result
        report["status"] = (
            "COMPLETE" if set(args.arms) <= report["results"].keys() else "PARTIAL"
        )
        (args.out / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
