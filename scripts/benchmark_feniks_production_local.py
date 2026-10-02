"""Synchronized production-asset benchmark using archived real observations.

This measures computation, not posterior convergence. No catalogue is recreated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps import model as model_module
from euclid_dsps import photometry_quadrature
from euclid_dsps.amortized.latent import x_to_theta
from euclid_dsps.amortized.posterior_target import (
    PosteriorObservation,
    posterior_log_target,
    safe_decoder_inputs,
)
from euclid_dsps.amortized.train import (
    _latent_spec_for_amortized_config,
    load_checkpoint,
)
from euclid_dsps.config import load_config
from euclid_dsps.filters import load_filters
from euclid_dsps.model import (
    DYNAMIC_CONTEXT_FIELDS,
    bind_dynamic_model_args,
    dynamic_model_args,
    load_context,
    predict_mags_jax,
    run_dsps_model_jax_dynamic,
)
from euclid_dsps.parameter_vectors import theta_vector_to_model_param_dict


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compile_function(fn, args):
    start = time.perf_counter()
    exe = jax.jit(fn).lower(*args).compile()
    seconds = time.perf_counter() - start
    memory = exe.memory_analysis()
    print(f"Compiled temporary bytes: {memory.temp_size_in_bytes}", flush=True)
    value = jax.block_until_ready(exe(*args))
    if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(value)):
        raise ValueError("Nonfinite benchmark result")
    for _ in range(2):
        jax.block_until_ready(exe(*args))
    return (
        exe,
        value,
        {
            "compile_seconds": seconds,
            "memory_bytes": {
                name: getattr(memory, name)
                for name in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                    "alias_size_in_bytes",
                )
            },
        },
    )


def measure(exe, args, repeats, metadata):
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(exe(*args))
        elapsed.append((time.perf_counter() - start) * 1000)
    return dict(
        metadata,
        median_ms=float(np.median(elapsed)),
        p95_ms=float(np.quantile(elapsed, 0.95)),
        raw_ms=elapsed,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--nuts-steps", type=int, default=3)
    parser.add_argument("--nuts-batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--compare-merge", action="store_true")
    parser.add_argument(
        "--implementation",
        choices=["current", "specialized", "configured"],
        default="current",
    )
    parser.add_argument("--verify-reference-chunk-size", type=int, default=0)
    parser.add_argument(
        "--merge-candidate",
        choices=["ordered", "dust_grouped", "trim_filters", "specialized"],
        default="ordered",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        choices=[
            "forward",
            "target_gradient",
            "likelihood_gradient",
            "guarded_prior_gradient",
            "prior_network_gradient",
        ],
    )
    args = parser.parse_args()
    if (
        args.out.exists()
        or args.repeats < 1
        or args.nuts_steps < 0
        or any(n < 1 or n > 64 for n in args.batches)
    ):
        parser.error("Fresh output, positive repeats, batch sizes 1..64 required")
    jax.config.update("jax_enable_x64", True)
    original = photometry_quadrature.merged_integral_jax
    if args.implementation == "specialized":
        from scripts.photometry_merge_candidate import specialized_integral

        photometry_quadrature.merged_integral_jax = specialized_integral
    try:
        run_measurements(args, *load_inputs(max(args.batches)))
    except Exception as error:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".failure.json").write_text(
            json.dumps(
                {
                    "status": "FAILED",
                    "requested_batches": args.batches,
                    "implementation": args.implementation,
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                    "successful_results_may_exist": args.out.exists(),
                },
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        photometry_quadrature.merged_integral_jax = original


def load_inputs(states: int):
    """Load verified assets and up to eight chain starts per archived object."""
    if not 1 <= states <= 64:
        raise ValueError("states must be in 1..64")
    root = Path("outputs/local_sampling_inputs")
    config_path = root / "reference/config.yaml"
    config = load_config(config_path)
    checkpoint = root / "checkpoints/best.eqx"
    manifest = json.loads((root / "reference/RUN_MANIFEST.json").read_text())
    for path, key in (
        (checkpoint, "checkpoint_sha256"),
        (config_path, "config_sha256"),
        (checkpoint.with_suffix(".eqx.json"), "checkpoint_sidecar_sha256"),
    ):
        if digest(path) != manifest["source"][key]:
            raise ValueError(f"Frozen reference hash differs: {path}")
    model = load_checkpoint(checkpoint, config)
    spec = _latent_spec_for_amortized_config(config)
    context = load_context(
        config["ssp_path"],
        load_filters(config["bands"]),
        n_sfh_bins=config["model"]["n_sfh_bins"],
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config["model"],
    )
    arrays = dynamic_model_args(context)
    archive = Path(
        "outputs/feniks_nuts_20260911/frozen_geometry_nuts_observed8_dense_depth6_v1"
    )
    first = Path(
        "outputs/feniks_nuts_20260911/frozen_geometry_nuts_dense_depth6_long_v1"
    )
    for source in (first, archive):
        archived_manifest = json.loads((source / "MANIFEST.json").read_text())
        inputs = archived_manifest["inputs"]
        for local in (
            config_path,
            checkpoint,
            root / "reference/observed_rows.npy",
            root / "reference/RUN_MANIFEST.json",
        ):
            if digest(local) not in inputs.values():
                raise ValueError(
                    f"Archived source provenance differs: {source}, {local}"
                )
    positions, observations, provenance = [], [], []
    for i in range(states):
        galaxy, chain = i % 8, i // 8
        directory = (first if galaxy == 0 else archive) / f"observed_{galaxy:03d}"
        with np.load(directory / "observation.npz") as obs:
            observations.append([obs[k][0] for k in ("flux", "flux_err", "mask")])
        starts = np.load(directory / "starts_B.npy")
        positions.append(starts[chain])
        provenance.append(
            {
                "directory": str(directory),
                "galaxy_case": galaxy,
                "archived_chain_start": chain,
                "observation_sha256": digest(directory / "observation.npz"),
                "starts_sha256": digest(directory / "starts_B.npy"),
            }
        )
    x = jnp.asarray(np.stack(positions), dtype=jnp.float64)
    flux, error, mask = [
        jnp.asarray(np.stack([o[k] for o in observations])) for k in range(3)
    ]
    return (
        config_path,
        config,
        checkpoint,
        model,
        spec,
        context,
        arrays,
        x,
        flux,
        error,
        mask,
        provenance,
    )


def run_measurements(
    args,
    config_path,
    config,
    checkpoint,
    model,
    spec,
    context,
    arrays,
    x,
    flux,
    error,
    mask,
    provenance,
):
    if args.implementation == "configured":
        context.model_config = {
            **context.model_config,
            "photometry_autodiff": "scalar_redshift_jvp_v1",
        }

    def parts(x, arrays, flux, error, mask):
        return posterior_log_target(
            model,
            x[None, None, :],
            PosteriorObservation(flux[None], error[None], mask[None]),
            spec,
            context,
            arrays,
            spec.names,
            config["amortized"]["likelihood"],
            {"calibration": config.get("calibration", {})},
        )

    def forward(*a):
        return parts(*a).model_flux[0, 0]

    def target(*a):
        return parts(*a).logtarget[0, 0]

    def likelihood(*a):
        return parts(*a).loglike[0, 0]

    def prior(*a):
        return parts(*a).logprior[0, 0]

    def prior_network(x, arrays, flux, error, mask):
        return model.prior.log_prob(x[None, :])[0]

    results = {}
    if args.profile_stages:

        def sed(position, arrays):
            safe_x, _ = safe_decoder_inputs(position, spec)
            theta = x_to_theta(safe_x, spec)
            params = theta_vector_to_model_param_dict(
                theta, spec.names, context.model_config
            )
            result = run_dsps_model_jax_dynamic(context, arrays, params)
            return result.dusted_rest_sed, params["z_obs"]

        sed_args = (x[0], arrays)
        executable, sed_value, metadata = compile_function(sed, sed_args)
        results["stage_sed"] = measure(executable, sed_args, args.repeats, metadata)

        def project(spectrum, z, arrays):
            bound = bind_dynamic_model_args(context, arrays)
            return predict_mags_jax(bound, bound.ssp_wave_jax, spectrum, z)

        projection_args = (*sed_value, arrays)
        for label, fn in (
            ("stage_projection", project),
            (
                "stage_projection_gradient",
                jax.value_and_grad(
                    lambda s, z, a: jnp.sum(project(s, z, a)), argnums=(0, 1)
                ),
            ),
        ):
            executable, _, metadata = compile_function(fn, projection_args)
            results[label] = measure(
                executable, projection_args, args.repeats, metadata
            )
        for key, result in results.items():
            print(key, json.dumps(result), flush=True)

    if args.compare_merge:
        from scripts.photometry_merge_candidate import (
            grouped_dust,
            ordered_integral,
            specialized_integral,
            trim_zero_filter_tails,
        )

        original = photometry_quadrature.merged_integral_jax
        original_dust = model_module.apply_popcosmos_dust_by_age_jax
        n = min(8, max(args.batches))
        paired_args = (x[:n], arrays, flux[:n], error[:n], mask[:n])
        paired_arguments = {"current": paired_args, "candidate": paired_args}
        if args.merge_candidate == "trim_filters":
            trimmed = list(arrays)
            index = DYNAMIC_CONTEXT_FIELDS.index("jax_filters")
            trimmed[index] = trim_zero_filter_tails(trimmed[index])
            paired_arguments["candidate"] = (
                x[:n],
                tuple(trimmed),
                flux[:n],
                error[:n],
                mask[:n],
            )
        executables, values, metadata = {}, {}, {}
        try:
            candidate = {
                "ordered": ordered_integral,
                "dust_grouped": original,
                "trim_filters": original,
                "specialized": specialized_integral,
            }[args.merge_candidate]
            for label, implementation in (
                ("current", original),
                ("candidate", candidate),
            ):
                photometry_quadrature.merged_integral_jax = implementation
                model_module.apply_popcosmos_dust_by_age_jax = (
                    grouped_dust(original_dust)
                    if label == "candidate" and "group" in args.merge_candidate
                    else original_dust
                )
                wrapped = jax.vmap(
                    jax.value_and_grad(target), in_axes=(0, None, 0, 0, 0)
                )
                executables[label], values[label], metadata[label] = compile_function(
                    wrapped, paired_arguments[label]
                )
            parity = []
            for old, new in zip(
                jax.tree.leaves(values["current"]),
                jax.tree.leaves(values["candidate"]),
                strict=True,
            ):
                np.testing.assert_allclose(old, new, rtol=1e-8, atol=1e-7)
                parity.append(float(np.max(np.abs(np.asarray(old) - np.asarray(new)))))
            elapsed = {label: [] for label in executables}
            for repeat in range(args.repeats):
                for label in (
                    ("current", "candidate")
                    if repeat % 2 == 0
                    else ("candidate", "current")
                ):
                    started = time.perf_counter()
                    jax.block_until_ready(executables[label](*paired_arguments[label]))
                    elapsed[label].append((time.perf_counter() - started) * 1000)
            for label, timings in elapsed.items():
                results[f"paired_merge_{label}"] = dict(
                    metadata[label],
                    median_ms=float(np.median(timings)),
                    p95_ms=float(np.quantile(timings, 0.95)),
                    raw_ms=timings,
                    states=n,
                    parity_max_abs=parity,
                    candidate=args.merge_candidate,
                )
                print(label, json.dumps(results[f"paired_merge_{label}"]), flush=True)
        finally:
            photometry_quadrature.merged_integral_jax = original
            model_module.apply_popcosmos_dust_by_age_jax = original_dust
    for n in args.batches:
        arguments = (x[:n], arrays, flux[:n], error[:n], mask[:n])
        for name, fn in (
            ("forward", forward),
            ("target_gradient", jax.value_and_grad(target)),
            ("likelihood_gradient", jax.value_and_grad(likelihood)),
            ("guarded_prior_gradient", jax.value_and_grad(prior)),
            ("prior_network_gradient", jax.value_and_grad(prior_network)),
        ):
            if args.components is not None and name not in args.components:
                continue
            print(f"Compiling {name} batch={n}", flush=True)
            exe, value, metadata = compile_function(
                jax.vmap(fn, in_axes=(0, None, 0, 0, 0)), arguments
            )
            result = measure(exe, arguments, args.repeats, metadata)
            result["evaluations_per_second"] = n * 1000 / result["median_ms"]
            result["values"] = [np.asarray(v).tolist() for v in jax.tree.leaves(value)]
            if name == "target_gradient" and args.verify_reference_chunk_size:
                current_impl = photometry_quadrature.merged_integral_jax
                from scripts.photometry_merge_candidate import reference_integral

                chunk = min(args.verify_reference_chunk_size, n)
                if n % chunk:
                    raise ValueError("Reference chunk size must divide state count")
                try:
                    photometry_quadrature.merged_integral_jax = reference_integral
                    saved_config = context.model_config
                    context.model_config = {
                        **saved_config,
                        "photometry_autodiff": "reverse_v1",
                    }
                    reference_fn = jax.vmap(
                        jax.value_and_grad(target), in_axes=(0, None, 0, 0, 0)
                    )
                    first_args = (
                        x[:chunk],
                        arrays,
                        flux[:chunk],
                        error[:chunk],
                        mask[:chunk],
                    )
                    reference_exe, _, _ = compile_function(reference_fn, first_args)
                    reference_values = []
                    for start in range(0, n, chunk):
                        stop = start + chunk
                        block = jax.block_until_ready(
                            reference_exe(
                                x[start:stop],
                                arrays,
                                flux[start:stop],
                                error[start:stop],
                                mask[start:stop],
                            )
                        )
                        reference_values.append(
                            tuple(np.asarray(v) for v in jax.tree.leaves(block))
                        )
                    maxima = []
                    for i, actual in enumerate(jax.tree.leaves(value)):
                        expected = np.concatenate([v[i] for v in reference_values])
                        np.testing.assert_allclose(
                            actual, expected, rtol=1e-8, atol=1e-7
                        )
                        maxima.append(
                            float(np.max(np.abs(np.asarray(actual) - expected)))
                        )
                    result["reference_microbatch_check"] = {
                        "status": "PASS",
                        "chunk_size": chunk,
                        "max_abs": maxima,
                    }
                finally:
                    photometry_quadrature.merged_integral_jax = current_impl
                    context.model_config = saved_config
            results[f"{name}_{n}"] = result
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k not in ("values", "raw_ms")}
                ),
                flush=True,
            )
        if args.nuts_steps and n in args.nuts_batches:

            def nuts(x, arrays, flux, error, mask, key):
                def logdensity(position):
                    return target(position, arrays, flux, error, mask)

                algorithm = blackjax.nuts(
                    logdensity,
                    step_size=0.001,
                    inverse_mass_matrix=jnp.ones(15, dtype=x.dtype),
                    max_num_doublings=4,
                )

                def step(state, key):
                    state, info = algorithm.step(key, state)
                    return state, (
                        info.num_integration_steps,
                        info.is_divergent,
                        info.acceptance_rate,
                    )

                return jax.lax.scan(
                    step, algorithm.init(x), jax.random.split(key, args.nuts_steps)
                )

            arguments = (*arguments, jax.random.split(jax.random.PRNGKey(261001), n))
            print(f"Compiling short NUTS batch={n}", flush=True)
            exe, value, metadata = compile_function(
                jax.vmap(nuts, in_axes=(0, None, 0, 0, 0, 0)), arguments
            )
            result = measure(exe, arguments, 3, metadata)
            result.update(
                transitions_per_target=args.nuts_steps,
                integration_steps=np.asarray(value[1][0]).tolist(),
                divergences=int(np.asarray(value[1][1]).sum()),
                acceptance=np.asarray(value[1][2]).tolist(),
            )
            results[f"nuts_{n}"] = result
            print(json.dumps(result), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "implementation": args.implementation,
            "jax_version": jax.__version__,
            "python_version": platform.python_version(),
            "blackjax_version": blackjax.__version__,
            "devices": [str(d) for d in jax.devices()],
            "device_kinds": [d.device_kind for d in jax.devices()],
            "config_sha256": digest(config_path),
            "checkpoint_sha256": digest(checkpoint),
            "bands": [b["name"] for b in config["bands"]],
            "model_config": config["model"],
            "runtime_model_config": context.model_config,
            "source_sha256": {
                path: digest(Path(path))
                for path in (
                    "scripts/benchmark_feniks_production_local.py",
                    "euclid_dsps/model.py",
                    "euclid_dsps/photometry_quadrature.py",
                )
            },
            "likelihood": config["amortized"]["likelihood"],
            "ssp_assets": {
                p: digest(p)
                for p in (config["ssp_path"], config["model"]["compressed_ssp_path"])
            },
            "dynamic_array_shapes": [
                {
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "bytes": v.size * v.dtype.itemsize,
                }
                for v in jax.tree.leaves(arrays)
            ],
            "prior_dtypes": sorted(
                set(
                    str(v.dtype)
                    for v in jax.tree.leaves(model.prior)
                    if hasattr(v, "dtype")
                )
            ),
            "observations": provenance,
            "results": results,
            "limitations": [
                "Up to eight observed galaxies and eight archived chain starts each; no catalogue reload.",
                "Loaded checkpoint precision, not a fully promoted float64 geometry model.",
                "Independent component graphs; times are not additive.",
                "NUTS fixed step .001, diagonal identity mass, depth4; no adaptation or convergence claim.",
                "One process per hardware; compilation excluded; no H100 measurement.",
            ],
        }
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
