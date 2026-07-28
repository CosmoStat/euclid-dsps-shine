#!/usr/bin/env python3
"""Run the FENIKS encoder/IS/MAP/NUTS/MCLMC posterior benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD", "0")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.amortized.data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from euclid_dsps.amortized.decoder import model_flux_from_x
from euclid_dsps.amortized.exact_posterior import (
    MCLMCSettings,
    NUTSSettings,
    TargetValues,
    combine_chain_diagnostics,
    normalized_importance_weights,
    run_adjusted_mclmc_chain,
    run_nuts_chain,
    run_unadjusted_mclmc_chain,
    systematic_resample,
)
from euclid_dsps.amortized.features import read_feature_stats
from euclid_dsps.amortized.latent import (
    latent_spec_hash,
    latent_spec_to_jsonable,
    theta_to_x,
    x_to_theta,
)
from euclid_dsps.amortized.likelihood import photometric_loglike
from euclid_dsps.amortized.map_adam import (
    _jit_latent_spec,
    _make_map_starts,
    _optimize_map_start_chunk_jit,
)
from euclid_dsps.amortized.posterior import sample_posterior
from euclid_dsps.amortized.train import (
    _latent_spec_for_amortized_config,
    _StaticArg,
    load_checkpoint,
)
from euclid_dsps.calibration import (
    apply_global_sed_scale_to_flux,
    apply_per_band_flux_calibration_to_flux,
    global_sed_scale_config,
    per_band_flux_calibration_config,
)
from euclid_dsps.config import load_config
from euclid_dsps.filters import load_filters
from euclid_dsps.io import ensure_dir
from euclid_dsps.model import (
    dynamic_model_args,
    load_context,
    predict_batch_seds,
)

try:
    from scripts.generate_feniks_individual_posteriors import select_representative_rows
except ModuleNotFoundError as error:
    if error.name not in {
        "scripts",
        "scripts.generate_feniks_individual_posteriors",
    }:
        raise
    from generate_feniks_individual_posteriors import select_representative_rows


@dataclass
class Runtime:
    config: dict[str, Any]
    model: Any
    latent_spec: Any
    context: Any
    model_args: Any
    filters: dict[str, Any]
    arrays: Any
    batch: Any
    likelihood: dict[str, Any]
    log_alpha_sed: jnp.ndarray
    log_alpha_band: jnp.ndarray
    use_global_scale: bool
    use_band_calibration: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare-cohort",
            "prepare-galaxy",
            "sample-chain",
            "finalize-mclmc",
            "finalize-galaxy",
            "finalize-run",
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--galaxy-index", type=int)
    parser.add_argument("--chain-index", type=int)
    parser.add_argument("--sampler", choices=("nuts", "mclmc", "mclmc_unadjusted"))
    parser.add_argument("--sampler-label")
    parser.add_argument("--pilot-selection", type=Path)
    parser.add_argument(
        "--cohort-file",
        type=Path,
        help="Optional real-data cohort with order, example_key, row_index, object_id.",
    )
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), default="full")
    parser.add_argument("--encoder-samples", type=int)
    parser.add_argument("--map-starts", type=int, default=16)
    parser.add_argument("--map-iterations", type=int)
    parser.add_argument("--nuts-warmup", type=int)
    parser.add_argument("--mclmc-tune", type=int)
    parser.add_argument("--sample-chunks")
    parser.add_argument("--thinning", type=int)
    parser.add_argument("--frac-tune", default="0.4,0.4,0.2")
    parser.add_argument("--desired-energy-var", type=float, default=5.0e-4)
    parser.add_argument("--seed", type=int, default=260727)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["catalog_path"] = str(args.dataset)
    if args.command == "prepare-cohort":
        prepare_cohort(args, config)
    elif args.command == "prepare-galaxy":
        prepare_galaxy(args, config)
    elif args.command == "sample-chain":
        sample_chain(args, config)
    elif args.command == "finalize-mclmc":
        finalize_mclmc(args, config)
    elif args.command == "finalize-galaxy":
        finalize_galaxy(args, config)
    else:
        finalize_run(args, config)


def prepare_cohort(args: argparse.Namespace, config: dict[str, Any]) -> None:
    out = ensure_dir(args.out)
    if args.cohort_file is not None:
        cohort = pd.read_csv(args.cohort_file)
        required = {"order", "example_key", "row_index", "object_id"}
        missing = sorted(required - set(cohort.columns))
        if missing:
            raise ValueError("Cohort file is missing columns: " + ", ".join(missing))
    else:
        cohort = select_representative_rows(config, args.dataset)
    if args.mode == "smoke":
        cohort = cohort.iloc[:2].reset_index(drop=True)
    cohort.to_parquet(out / "cohort.parquet", index=False)
    cohort.to_csv(out / "cohort.csv", index=False)
    contract = {
        "status": "prepared",
        "config": str(args.config),
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "feature_stats": str(args.feature_stats),
        "config_sha256": _sha256(args.config),
        "dataset_sha256": _sha256(args.dataset),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_sidecar_sha256": _sha256(
            args.checkpoint.with_suffix(args.checkpoint.suffix + ".json")
        ),
        "feature_stats_sha256": _sha256(args.feature_stats),
        "code_commit": _git_commit(),
        "selected_rows": cohort["row_index"].astype(int).tolist(),
        "selection_labels": cohort["example_key"].astype(str).tolist(),
        "mode": args.mode,
        "likelihood": {
            "type": config["amortized"]["likelihood"]["type"],
            "student_t_dof": config["amortized"]["likelihood"]["student_t_dof"],
            "error_floor_frac": config["amortized"]["likelihood"][
                "error_floor_frac"
            ],
            "error_jitter": config["amortized"]["likelihood"]["error_jitter"],
        },
    }
    _write_json(out / "contract.json", contract)
    (out / "COHORT_DONE").touch()
    print(json.dumps(contract, indent=2))


def prepare_galaxy(args: argparse.Namespace, config: dict[str, Any]) -> None:
    galaxy_index = _required_index(args.galaxy_index, "galaxy-index")
    item = _cohort_item(args.out, galaxy_index)
    galaxy_dir = _galaxy_dir(args.out, item)
    done = galaxy_dir / "PREP_DONE"
    if done.exists() and not args.overwrite:
        print(f"[exact-benchmark] preparation already complete: {galaxy_dir}")
        return
    runtime = _load_runtime(args, config, int(item.row_index))
    n_encoder = int(
        args.encoder_samples
        if args.encoder_samples is not None
        else (256 if args.mode == "smoke" else 32_768)
    )
    map_iterations = int(
        args.map_iterations
        if args.map_iterations is not None
        else (20 if args.mode == "smoke" else 2_000)
    )
    galaxy_dir.mkdir(parents=True, exist_ok=True)
    _write_observation(runtime, galaxy_dir, item)
    key = jax.random.PRNGKey(int(args.seed) + galaxy_index * 10_000)
    key, encoder_key = jax.random.split(key)
    posterior = sample_posterior(
        runtime.model,
        encoder_key,
        runtime.batch.features,
        n_encoder,
    )
    x = posterior.x[:, 0, :]
    logq = posterior.logq[:, 0]
    target_fn = _target_components_fn(runtime)
    target = _evaluate_target_chunks(target_fn, x, chunk_size=256)
    theta = np.asarray(jax.device_get(x_to_theta(x, runtime.latent_spec)))
    encoder = _sample_frame(
        np.asarray(jax.device_get(x)),
        theta,
        runtime.latent_spec.names,
        logq=np.asarray(jax.device_get(logq)),
        target=target,
    )
    _write_parquet(encoder, galaxy_dir / "encoder_samples.parquet")
    importance = normalized_importance_weights(
        encoder["logtarget"].to_numpy(),
        encoder["logq"].to_numpy(),
    )
    weighted = encoder.copy()
    for name in ("log_weight", "weight", "psis_weight"):
        weighted[name] = importance[name]
    _write_parquet(weighted, galaxy_dir / "importance_weighted_samples.parquet")
    resample_index = systematic_resample(
        np.asarray(importance["psis_weight"]),
        min(n_encoder, 8_192),
        seed=int(args.seed) + galaxy_index,
    )
    resampled = weighted.iloc[resample_index].reset_index(drop=True)
    resampled.insert(0, "resample_draw", np.arange(len(resampled)))
    _write_parquet(resampled, galaxy_dir / "importance_resampled_samples.parquet")
    _write_json(
        galaxy_dir / "importance_diagnostics.json",
        {
            key_name: (
                float(value) if np.isfinite(float(value)) else None
            )
            for key_name, value in importance.items()
            if np.isscalar(value)
        },
    )
    key, map_key = jax.random.split(key)
    map_frame, map_trace, best_x = _run_map(
        runtime,
        map_key,
        n_starts=int(args.map_starts),
        maxiter=map_iterations,
    )
    _write_parquet(map_frame, galaxy_dir / "map_solutions.parquet")
    _write_parquet(map_trace, galaxy_dir / "map_trace.parquet")
    initial = _select_chain_starts(
        best_x,
        encoder,
        n_chains=4,
    )
    np.save(galaxy_dir / "initial_positions.npy", initial)
    truth_theta = _truth_theta(runtime)
    audit_parts = [
        np.median(np.asarray(jax.device_get(x)), axis=0)[None, :],
        initial,
    ]
    audit_labels = ["encoder_median", *[f"chain_start_{i}" for i in range(4)]]
    if truth_theta is not None:
        truth_x = np.asarray(
            jax.device_get(theta_to_x(jnp.asarray(truth_theta), runtime.latent_spec))
        )
        truth_target = target_fn(jnp.asarray(truth_x))
        truth = pd.DataFrame(
            [
                {
                    **{
                        f"x_{name}": truth_x[i]
                        for i, name in enumerate(runtime.latent_spec.names)
                    },
                    **{
                        name: truth_theta[i]
                        for i, name in enumerate(runtime.latent_spec.names)
                    },
                    "loglike": float(truth_target.loglike),
                    "logprior": float(truth_target.logprior),
                    "logtarget": float(truth_target.logtarget),
                }
            ]
        )
        _write_parquet(truth, galaxy_dir / "truth.parquet")
        audit_parts.append(truth_x[None, :])
        audit_labels.append("truth")
    audit_points = np.concatenate(audit_parts, axis=0)
    audit = _evaluate_target_chunks(target_fn, jnp.asarray(audit_points), chunk_size=8)
    logdensity = _logdensity_fn(runtime)
    gradients = np.asarray(
        jax.device_get(jax.jit(jax.vmap(jax.grad(logdensity)))(jnp.asarray(audit_points)))
    )
    roundtrip = np.asarray(
        jax.device_get(
            theta_to_x(
                x_to_theta(jnp.asarray(audit_points), runtime.latent_spec),
                runtime.latent_spec,
            )
        )
    )
    gradient_norm = np.linalg.norm(gradients, axis=1)
    audit_frame = pd.DataFrame(
        {
            "point": audit_labels,
            "loglike": audit.loglike,
            "logprior": audit.logprior,
            "logtarget": audit.logtarget,
            "gradient_norm": gradient_norm,
            "roundtrip_max_abs_error": np.max(
                np.abs(roundtrip - audit_points), axis=1
            ),
        }
    )
    for index, name in enumerate(runtime.latent_spec.names):
        audit_frame[f"grad_{name}"] = gradients[:, index]
        audit_frame[f"normalized_abs_grad_{name}"] = np.abs(
            gradients[:, index]
        ) / np.maximum(gradient_norm, 1.0e-12)
    _write_parquet(audit_frame, galaxy_dir / "target_audit.parquet")
    finite_audit = bool(
        np.isfinite(audit_frame.select_dtypes(include=[np.number])).all().all()
    )
    roundtrip_error = float(audit_frame["roundtrip_max_abs_error"].max())
    if not finite_audit or roundtrip_error > 2.0e-5:
        raise RuntimeError(
            "Target audit failed: "
            f"finite={finite_audit} roundtrip_max_abs_error={roundtrip_error}"
        )
    manifest = {
        "galaxy_index": galaxy_index,
        "row_index": int(item.row_index),
        "object_id": str(item.object_id),
        "example_key": str(item.example_key),
        "encoder_samples": n_encoder,
        "map_starts": int(args.map_starts),
        "map_iterations": map_iterations,
        "normalization_hash": latent_spec_hash(runtime.latent_spec),
        "latent_spec": latent_spec_to_jsonable(runtime.latent_spec),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "target_audit_finite": finite_audit,
        "roundtrip_max_abs_error": roundtrip_error,
        "truth_available": truth_theta is not None,
    }
    _write_json(galaxy_dir / "prepare_manifest.json", manifest)
    done.touch()
    print(json.dumps(manifest, indent=2))


def sample_chain(args: argparse.Namespace, config: dict[str, Any]) -> None:
    galaxy_index = _required_index(args.galaxy_index, "galaxy-index")
    chain_index = _required_index(args.chain_index, "chain-index")
    if not args.sampler:
        raise ValueError("--sampler is required for sample-chain")
    print(
        "[exact-chain] "
        f"sampler={args.sampler} galaxy={galaxy_index} chain={chain_index} "
        f"mode={args.mode}",
        flush=True,
    )
    item = _cohort_item(args.out, galaxy_index)
    galaxy_dir = _galaxy_dir(args.out, item)
    if not (galaxy_dir / "PREP_DONE").exists():
        raise FileNotFoundError(f"Galaxy preparation is incomplete: {galaxy_dir}")
    print(f"[exact-chain] loading runtime for row={int(item.row_index)}", flush=True)
    runtime = _load_runtime(args, config, int(item.row_index))
    print("[exact-chain] runtime ready; constructing target", flush=True)
    logdensity_fn = _logdensity_fn(runtime)
    initial = np.load(galaxy_dir / "initial_positions.npy")
    if chain_index >= len(initial):
        raise IndexError(f"chain-index {chain_index} exceeds {len(initial)} starts")
    chunks = _sample_chunks(args)
    sampler_label = str(args.sampler_label or args.sampler)
    chain_dir = galaxy_dir / sampler_label / f"chain_{chain_index:02d}"
    seed = int(args.seed) + galaxy_index * 10_000 + chain_index * 101
    if args.sampler == "nuts":
        settings = NUTSSettings(
            warmup_steps=int(
                args.nuts_warmup
                if args.nuts_warmup is not None
                else (10 if args.mode == "smoke" else 500)
            ),
            sample_chunks=chunks,
            target_accept=0.65,
            max_num_doublings=4 if args.mode == "smoke" else 10,
        )
        print(
            f"[exact-chain] NUTS warmup={settings.warmup_steps} "
            f"chunks={settings.sample_chunks}",
            flush=True,
        )
        manifest = run_nuts_chain(
            logdensity_fn,
            jnp.asarray(initial[chain_index]),
            seed=seed,
            settings=settings,
            out_dir=chain_dir,
        )
    else:
        frac_text = args.frac_tune
        thinning = args.thinning
        if args.pilot_selection is not None:
            selection = json.loads(args.pilot_selection.read_text(encoding="utf-8"))
            frac_text = ",".join(str(value) for value in selection["frac_tune"])
            thinning = int(selection["thinning"])
        frac = tuple(float(value) for value in frac_text.split(","))
        if len(frac) != 3:
            raise ValueError("--frac-tune must contain three comma-separated values")
        settings = MCLMCSettings(
            tune_steps=int(
                args.mclmc_tune
                if args.mclmc_tune is not None
                else (0 if args.mode == "smoke" else 500)
            ),
            sample_chunks=chunks,
            thinning=int(
                thinning if thinning is not None else (1 if args.mode == "smoke" else 8)
            ),
            target_accept=0.8,
            initial_step_size=5.0e-2 if args.mode == "smoke" else 1.0e-3,
            frac_tune1=frac[0],
            frac_tune2=frac[1],
            frac_tune3=frac[2],
            diagonal_preconditioning=True,
            collapse_ratio=1.0e-4,
            desired_energy_var=float(args.desired_energy_var),
        )
        runner = (
            run_unadjusted_mclmc_chain
            if args.sampler == "mclmc_unadjusted"
            else run_adjusted_mclmc_chain
        )
        print(
            f"[exact-chain] {args.sampler} tune={settings.tune_steps} "
            f"thinning={settings.thinning} chunks={settings.sample_chunks}",
            flush=True,
        )
        manifest = runner(
            logdensity_fn,
            jnp.asarray(initial[chain_index]),
            seed=seed,
            settings=settings,
            out_dir=chain_dir,
        )
    (chain_dir / "DONE").touch()
    print(json.dumps(manifest, indent=2))


def finalize_galaxy(args: argparse.Namespace, config: dict[str, Any]) -> None:
    galaxy_index = _required_index(args.galaxy_index, "galaxy-index")
    item = _cohort_item(args.out, galaxy_index)
    galaxy_dir = _galaxy_dir(args.out, item)
    runtime = _load_runtime(args, config, int(item.row_index))
    for sampler in ("nuts", "mclmc"):
        n_chains = 2 if args.mode == "smoke" and sampler == "mclmc" else 4
        directories = [
            galaxy_dir / sampler / f"chain_{i:02d}" for i in range(n_chains)
        ]
        missing = [path for path in directories if not (path / "DONE").exists()]
        if missing:
            raise FileNotFoundError(
                f"{sampler} has incomplete chains: " + ", ".join(map(str, missing))
            )
        diagnostics, summary = combine_chain_diagnostics(
            directories,
            parameter_names=runtime.latent_spec.names,
        )
        _write_parquet(diagnostics, galaxy_dir / sampler / "diagnostics.parquet")
        _write_json(galaxy_dir / sampler / "diagnostics.json", summary)
        samples = _combine_physical_chains(directories, runtime.latent_spec)
        _write_parquet(samples, galaxy_dir / sampler / "samples.parquet")
    if args.mode != "smoke":
        _write_corner_plots(galaxy_dir, runtime, item)
        _write_sed_and_photometry(galaxy_dir, runtime, item)
        _write_convergence_plots(galaxy_dir, runtime.latent_spec.names)
    else:
        _write_json(
            galaxy_dir / "smoke_summary.json",
            {
                "status": "passed",
                "plots_skipped": True,
                "reason": "smoke chains are intentionally too short for inference plots",
            },
        )
    (galaxy_dir / "DONE").touch()


def finalize_mclmc(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Combine an adjusted-MCLMC-only real-data run without requiring truth."""
    galaxy_index = _required_index(args.galaxy_index, "galaxy-index")
    item = _cohort_item(args.out, galaxy_index)
    galaxy_dir = _galaxy_dir(args.out, item)
    runtime = _load_runtime(args, config, int(item.row_index))
    n_chains = 2 if args.mode == "smoke" else 4
    directories = [
        galaxy_dir / "mclmc" / f"chain_{index:02d}" for index in range(n_chains)
    ]
    missing = [path for path in directories if not (path / "DONE").exists()]
    if missing:
        raise FileNotFoundError(
            "MCLMC has incomplete chains: " + ", ".join(map(str, missing))
        )
    diagnostics, summary = combine_chain_diagnostics(
        directories, parameter_names=runtime.latent_spec.names
    )
    _write_parquet(diagnostics, galaxy_dir / "mclmc/diagnostics.parquet")
    _write_json(galaxy_dir / "mclmc/diagnostics.json", summary)
    samples = _combine_physical_chains(directories, runtime.latent_spec)
    _write_parquet(samples, galaxy_dir / "mclmc/samples.parquet")
    _write_json(
        galaxy_dir / "mclmc_real_data_summary.json",
        {
            "status": "complete",
            "truth_available": False,
            "sampler": "adjusted_mclmc",
            "n_chains": n_chains,
            "n_samples": int(len(samples)),
            "diagnostics": summary,
        },
    )
    (galaxy_dir / "MCLMC_DONE").touch()


def finalize_run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    cohort = pd.read_parquet(args.out / "cohort.parquet")
    rows = []
    for index, item in enumerate(cohort.itertuples(index=False)):
        galaxy_dir = _galaxy_dir(args.out, item)
        if not (galaxy_dir / "DONE").exists():
            raise FileNotFoundError(f"Incomplete galaxy {index}: {galaxy_dir}")
        row = {
            "galaxy_index": index,
            "example_key": item.example_key,
            "row_index": int(item.row_index),
        }
        for sampler in ("nuts", "mclmc"):
            diagnostics = json.loads(
                (galaxy_dir / sampler / "diagnostics.json").read_text()
            )
            row[f"{sampler}_max_rhat"] = diagnostics["max_rhat"]
            row[f"{sampler}_min_bulk_ess"] = diagnostics["min_bulk_ess"]
            row[f"{sampler}_min_tail_ess"] = diagnostics["min_tail_ess"]
        importance = json.loads(
            (galaxy_dir / "importance_diagnostics.json").read_text()
        )
        row["importance_pareto_k"] = importance["pareto_k"]
        row["importance_raw_ess"] = importance["raw_ess"]
        rows.append(row)
    scoreboard = pd.DataFrame(rows)
    _write_parquet(scoreboard, args.out / "scoreboard.parquet")
    scoreboard.to_csv(args.out / "scoreboard.csv", index=False)
    contract = json.loads((args.out / "contract.json").read_text(encoding="utf-8"))
    if contract.get("mode") != "smoke":
        _write_run_comparison(args.out, cohort)
    _write_json(
        args.out / "benchmark_summary.json",
        {
            "galaxies": int(len(scoreboard)),
            "methods": ["encoder", "importance", "map", "nuts", "mclmc", "truth"],
            "all_nuts_rhat_pass": bool((scoreboard["nuts_max_rhat"] <= 1.01).all()),
            "all_mclmc_rhat_pass": bool(
                (scoreboard["mclmc_max_rhat"] <= 1.01).all()
            ),
            "code_commit": _git_commit(),
        },
    )
    (args.out / "DONE").touch()


def _write_run_comparison(out: Path, cohort: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from scipy.stats import wasserstein_distance

    posterior_rows = []
    photometry_rows = []
    methods = {
        "Encoder": "encoder_samples.parquet",
        "Encoder + IS": "importance_resampled_samples.parquet",
        "NUTS": "nuts/samples.parquet",
        "MCLMC": "mclmc/samples.parquet",
    }
    for item in cohort.itertuples(index=False):
        galaxy_dir = _galaxy_dir(out, item)
        prepare = json.loads(
            (galaxy_dir / "prepare_manifest.json").read_text(encoding="utf-8")
        )
        names = tuple(prepare["latent_spec"]["names"])
        truth = pd.read_parquet(galaxy_dir / "truth.parquet").iloc[0]
        map_best = (
            pd.read_parquet(galaxy_dir / "map_solutions.parquet")
            .sort_values("objective")
            .iloc[0]
        )
        frames = {
            label: pd.read_parquet(galaxy_dir / relative)
            for label, relative in methods.items()
        }
        reference = frames["NUTS"]
        for name in names:
            ref = reference[name].to_numpy(dtype=float)
            ref_mean = float(np.mean(ref))
            ref_std = float(np.std(ref, ddof=1))
            scale = max(ref_std, 1.0e-8)
            for label, frame in frames.items():
                values = frame[name].to_numpy(dtype=float)
                posterior_rows.append(
                    {
                        "example_key": item.example_key,
                        "row_index": int(item.row_index),
                        "parameter": name,
                        "method": label,
                        "mean": np.mean(values),
                        "std": np.std(values, ddof=1),
                        "nuts_standardized_mean_offset": (
                            (np.mean(values) - ref_mean) / scale
                        ),
                        "std_ratio_to_nuts": np.std(values, ddof=1) / scale,
                        "truth_z_score": (np.mean(values) - truth[name]) / scale,
                        "wasserstein_to_nuts_in_nuts_std": (
                            wasserstein_distance(values, ref) / scale
                        ),
                    }
                )
            posterior_rows.append(
                {
                    "example_key": item.example_key,
                    "row_index": int(item.row_index),
                    "parameter": name,
                    "method": "MAP",
                    "mean": map_best[name],
                    "std": np.nan,
                    "nuts_standardized_mean_offset": (
                        (map_best[name] - ref_mean) / scale
                    ),
                    "std_ratio_to_nuts": np.nan,
                    "truth_z_score": (map_best[name] - truth[name]) / scale,
                    "wasserstein_to_nuts_in_nuts_std": np.nan,
                }
            )
        observation = pd.read_parquet(galaxy_dir / "observation.parquet")
        predictions = pd.read_parquet(
            galaxy_dir / "photometric_predictions.parquet"
        )
        for label, frame in predictions.groupby("method"):
            merged = observation.merge(frame, on="band", validate="one_to_one")
            valid = merged["mask"].astype(bool) & (merged["flux_err"] > 0)
            residual = (
                merged.loc[valid, "flux_q50"] - merged.loc[valid, "flux"]
            ) / merged.loc[valid, "flux_err"]
            photometry_rows.append(
                {
                    "example_key": item.example_key,
                    "row_index": int(item.row_index),
                    "method": label,
                    "chi2_median_prediction": float(np.sum(residual**2)),
                    "residual_rms": float(np.sqrt(np.mean(residual**2))),
                    "n_bands": int(valid.sum()),
                }
            )
    posterior = pd.DataFrame(posterior_rows)
    photometry = pd.DataFrame(photometry_rows)
    _write_parquet(posterior, out / "posterior_agreement.parquet")
    posterior.to_csv(out / "posterior_agreement.csv", index=False)
    _write_parquet(photometry, out / "photometric_fit_metrics.parquet")
    photometry.to_csv(out / "photometric_fit_metrics.csv", index=False)

    names = list(dict.fromkeys(posterior["parameter"]))
    shown_methods = ["Encoder", "Encoder + IS", "MCLMC", "MAP"]
    metrics = (
        (
            "nuts_standardized_mean_offset",
            "Median mean offset from NUTS [NUTS std]",
            "coolwarm",
            (-2.0, 2.0),
        ),
        (
            "std_ratio_to_nuts",
            "Median posterior std / NUTS std",
            "viridis",
            (0.0, 2.0),
        ),
        (
            "truth_z_score",
            "Median posterior mean - truth [NUTS std]",
            "coolwarm",
            (-2.0, 2.0),
        ),
    )
    fig, axes = plt.subplots(3, 1, figsize=(15, 9.5), constrained_layout=True)
    for ax, (column, title, cmap, limits) in zip(axes, metrics, strict=True):
        matrix = np.full((len(shown_methods), len(names)), np.nan)
        for row, method in enumerate(shown_methods):
            subset = posterior.loc[posterior["method"] == method]
            for col, name in enumerate(names):
                values = subset.loc[
                    subset["parameter"] == name, column
                ].to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                matrix[row, col] = (
                    np.median(finite) if finite.size else np.nan
                )
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
        )
        ax.set_yticks(range(len(shown_methods)), labels=shown_methods)
        ax.set_xticks(range(len(names)), labels=names, rotation=55, ha="right")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, pad=0.01)
    fig.savefig(out / "posterior_method_agreement.png", dpi=190)
    fig.savefig(out / "posterior_method_agreement.pdf")
    plt.close(fig)

    order = ["Truth", "MAP", "Encoder", "Encoder + IS", "NUTS", "MCLMC"]
    summary = (
        photometry.groupby("method")["chi2_median_prediction"]
        .agg(["median", "min", "max"])
        .reindex([name for name in order if name in photometry["method"].unique()])
    )
    summary.to_csv(out / "photometric_fit_summary.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(
        np.arange(len(summary)),
        summary["median"],
        yerr=np.vstack(
            [
                summary["median"] - summary["min"],
                summary["max"] - summary["median"],
            ]
        ),
        capsize=3,
        color="#0072B2",
    )
    ax.set_xticks(
        np.arange(len(summary)), labels=summary.index, rotation=25, ha="right"
    )
    ax.set_ylabel("Photometric chi2 of median prediction")
    ax.set_title("Same seven galaxies and same DSPS likelihood inputs")
    fig.tight_layout()
    fig.savefig(out / "photometric_fit_comparison.png", dpi=190)
    fig.savefig(out / "photometric_fit_comparison.pdf")
    plt.close(fig)


def _load_runtime(
    args: argparse.Namespace,
    config: dict[str, Any],
    row_index: int,
) -> Runtime:
    feature_stats = read_feature_stats(args.feature_stats)
    arrays = load_photometry_arrays_from_config(
        config,
        batch_size=10_000,
        row_indices=np.asarray([row_index], dtype=np.int64),
    )
    batch = next(
        iter(
            iter_photometry_batches_from_arrays(
                arrays,
                batch_size=1,
                feature_stats=feature_stats,
            )
        )
    )
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    model = load_checkpoint(args.checkpoint, config)
    latent_spec = _latent_spec_for_amortized_config(config)
    scale_cfg = global_sed_scale_config(
        {"calibration": config.get("calibration", {}) or {}}
    )
    band_cfg = per_band_flux_calibration_config(
        {"calibration": config.get("calibration", {}) or {}}
    )
    log_alpha_sed = (
        model.sed_scale.log_alpha_sed
        if scale_cfg.enabled
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    log_alpha_band = (
        model.band_calibration.log_alpha_band
        if band_cfg.enabled and model.band_calibration is not None
        else jnp.zeros((len(config["bands"]),), dtype=jnp.float32)
    )
    return Runtime(
        config=config,
        model=model,
        latent_spec=latent_spec,
        context=context,
        model_args=dynamic_model_args(context),
        filters=filters,
        arrays=arrays,
        batch=batch,
        likelihood=config["amortized"]["likelihood"],
        log_alpha_sed=log_alpha_sed,
        log_alpha_band=log_alpha_band,
        use_global_scale=bool(scale_cfg.enabled),
        use_band_calibration=bool(band_cfg.enabled),
    )


def _target_components_fn(runtime: Runtime):
    def components(x):
        model_flux = model_flux_from_x(
            x[None, None, :],
            runtime.latent_spec,
            runtime.context,
            runtime.model_args,
            runtime.latent_spec.names,
        )
        if runtime.use_global_scale:
            model_flux = apply_global_sed_scale_to_flux(
                model_flux, runtime.log_alpha_sed
            )
        if runtime.use_band_calibration:
            model_flux = apply_per_band_flux_calibration_to_flux(
                model_flux, runtime.log_alpha_band
            )
        loglike = photometric_loglike(
            runtime.batch.flux,
            model_flux,
            runtime.batch.flux_err,
            runtime.batch.mask,
            likelihood_type=str(runtime.likelihood["type"]),
            student_t_dof=float(runtime.likelihood["student_t_dof"]),
            error_floor_frac=float(runtime.likelihood["error_floor_frac"]),
            error_jitter=float(runtime.likelihood["error_jitter"]),
        )[0, 0]
        logprior = runtime.model.prior.log_prob(x)
        return TargetValues(loglike, logprior, loglike + logprior)

    return components


def _logdensity_fn(runtime: Runtime):
    components = _target_components_fn(runtime)

    def logdensity(x):
        return components(x).logtarget

    return logdensity


def _evaluate_target_chunks(target_fn, x, *, chunk_size: int) -> TargetValues:
    arrays = []
    vectorized = jax.jit(jax.vmap(target_fn))
    for start in range(0, int(x.shape[0]), int(chunk_size)):
        values = vectorized(x[start : start + int(chunk_size)])
        arrays.append(jax.device_get(values))
    return TargetValues(
        *[
            np.concatenate([np.asarray(value[index]) for value in arrays], axis=0)
            for index in range(3)
        ]
    )


def _run_map(runtime: Runtime, key, *, n_starts: int, maxiter: int):
    mean, log_std = runtime.model.encoder(runtime.batch.features)
    starts, families = _make_map_starts(
        runtime.model,
        mean,
        log_std,
        runtime.latent_spec,
        key,
        n_starts=n_starts,
        start_mode="mixed",
    )
    result = _optimize_map_start_chunk_jit(
        runtime.model,
        runtime.batch.flux,
        runtime.batch.flux_err,
        runtime.batch.mask,
        _jit_latent_spec(runtime.latent_spec),
        _StaticArg(runtime.context),
        runtime.model_args,
        runtime.latent_spec.names,
        starts,
        maxiter=maxiter,
        learning_rate=2.0e-3,
        prior_weight=1.0,
        prior_density_space="x",
        likelihood_type=str(runtime.likelihood["type"]),
        student_t_dof=float(runtime.likelihood["student_t_dof"]),
        error_floor_frac=float(runtime.likelihood["error_floor_frac"]),
        error_jitter=float(runtime.likelihood["error_jitter"]),
        log_alpha_sed=runtime.log_alpha_sed,
        log_alpha_band=runtime.log_alpha_band,
        use_global_scale=runtime.use_global_scale,
        use_band_calibration=runtime.use_band_calibration,
    )
    x = np.asarray(jax.device_get(result["best_x"]))[:, 0, :]
    theta = np.asarray(jax.device_get(x_to_theta(result["best_x"], runtime.latent_spec)))[
        :, 0, :
    ]
    objective = np.asarray(jax.device_get(result["best_objective"]))[:, 0]
    rows = {
        "start": np.arange(n_starts),
        "start_family": list(families),
        "objective": objective,
        "photometric_nll": np.asarray(jax.device_get(result["best_nll"]))[:, 0],
        "logprior": np.asarray(jax.device_get(result["best_logprior"]))[:, 0],
        "chi2": np.asarray(jax.device_get(result["best_chi2"]))[:, 0],
        "grad_norm": np.asarray(jax.device_get(result["grad_norm"]))[:, 0],
    }
    for index, name in enumerate(runtime.latent_spec.names):
        rows[f"x_{name}"] = x[:, index]
        rows[name] = theta[:, index]
    trace = np.asarray(jax.device_get(result["trace_objective"]))
    trace_frame = pd.DataFrame(
        {"iteration": np.arange(len(trace)), "mean_objective": trace}
    )
    best = int(np.argmin(objective))
    return pd.DataFrame(rows), trace_frame, x[best]


def _select_chain_starts(
    best_x: np.ndarray,
    encoder: pd.DataFrame,
    *,
    n_chains: int,
) -> np.ndarray:
    x_columns = [column for column in encoder if column.startswith("x_")]
    candidates = encoder.sort_values("logtarget", ascending=False)
    selected = [np.asarray(best_x, dtype=np.float32)]
    pool = candidates[x_columns].to_numpy(dtype=np.float32)
    for _ in range(1, int(n_chains)):
        distance = np.min(
            np.stack(
                [np.sum((pool - value[None, :]) ** 2, axis=1) for value in selected]
            ),
            axis=0,
        )
        selected.append(pool[int(np.argmax(distance))])
    return np.stack(selected)


def _sample_frame(x, theta, names, *, logq, target) -> pd.DataFrame:
    data = {
        "draw": np.arange(len(x), dtype=np.int64),
        "logq": np.asarray(logq),
        "loglike": np.asarray(target.loglike),
        "logprior": np.asarray(target.logprior),
        "logtarget": np.asarray(target.logtarget),
    }
    for index, name in enumerate(names):
        data[f"x_{name}"] = x[:, index]
        data[name] = theta[:, index]
    return pd.DataFrame(data)


def _truth_theta(runtime: Runtime) -> np.ndarray | None:
    if runtime.arrays.truth is None:
        return None
    missing = [
        name for name in runtime.latent_spec.names if name not in runtime.arrays.truth
    ]
    if missing:
        raise ValueError("Missing truth parameters: " + ", ".join(missing))
    return np.asarray(
        [runtime.arrays.truth[name][0] for name in runtime.latent_spec.names],
        dtype=np.float32,
    )


def _write_observation(runtime: Runtime, galaxy_dir: Path, item) -> None:
    frame = pd.DataFrame(
        {
            "band": list(runtime.arrays.band_names),
            "flux": runtime.arrays.flux[0],
            "flux_err": runtime.arrays.flux_err[0],
            "mask": runtime.arrays.mask[0],
            "row_index": int(item.row_index),
            "object_id": str(item.object_id),
        }
    )
    _write_parquet(frame, galaxy_dir / "observation.parquet")


def _combine_physical_chains(directories, latent_spec) -> pd.DataFrame:
    pieces = []
    x_columns = [f"x_{index:02d}" for index in range(len(latent_spec.names))]
    for chain, directory in enumerate(directories):
        paths = sorted(
            path
            for path in (Path(directory) / "chunks").glob("part_*.parquet")
            if not path.name.endswith("_info.parquet")
        )
        offset = 0
        for chunk, path in enumerate(paths):
            frame = pd.read_parquet(path)
            x = frame[x_columns].to_numpy(dtype=np.float32)
            theta = np.asarray(jax.device_get(x_to_theta(x, latent_spec)))
            output = pd.DataFrame(
                {
                    "chain": chain,
                    "chunk": chunk,
                    "draw": np.arange(offset, offset + len(frame)),
                }
            )
            for index, name in enumerate(latent_spec.names):
                output[f"x_{name}"] = x[:, index]
                output[name] = theta[:, index]
            pieces.append(output)
            offset += len(frame)
    return pd.concat(pieces, ignore_index=True)


def _write_corner_plots(galaxy_dir: Path, runtime: Runtime, item) -> None:
    import corner
    import matplotlib.pyplot as plt

    names = runtime.latent_spec.names
    methods = {
        "Encoder": pd.read_parquet(galaxy_dir / "encoder_samples.parquet"),
        "Encoder + IS": pd.read_parquet(
            galaxy_dir / "importance_resampled_samples.parquet"
        ),
        "NUTS": pd.read_parquet(galaxy_dir / "nuts" / "samples.parquet"),
        "MCLMC": pd.read_parquet(galaxy_dir / "mclmc" / "samples.parquet"),
    }
    truth = pd.read_parquet(galaxy_dir / "truth.parquet").iloc[0]
    map_best = (
        pd.read_parquet(galaxy_dir / "map_solutions.parquet")
        .sort_values("objective")
        .iloc[0]
    )
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    for subset_name, subset in (
        ("corner_key5", ("z_obs", "log10_stellar_mass", "dust_av", "dust_delta", "sfh_dlog_sfr_01")),
        ("corner_full15", names),
    ):
        labels = list(subset)
        figure = None
        for (label, frame), color in zip(methods.items(), colors, strict=True):
            values = frame[labels].to_numpy(dtype=float)
            if len(values) > 2_000:
                values = values[np.linspace(0, len(values) - 1, 2_000).astype(int)]
            figure = corner.corner(
                values,
                labels=labels,
                fig=figure,
                color=color,
                plot_datapoints=False,
                fill_contours=False,
                levels=(0.5, 0.9),
                hist_kwargs={"density": True, "label": label},
                contour_kwargs={"linewidths": 1.0},
                quiet=True,
            )
        ndim = len(labels)
        axes = np.asarray(figure.axes).reshape((ndim, ndim))
        truth_values = truth[labels].to_numpy(dtype=float)
        map_values = map_best[labels].to_numpy(dtype=float)
        for row in range(ndim):
            axes[row, row].axvline(truth_values[row], color="black", lw=1.1)
            axes[row, row].axvline(map_values[row], color="#E69F00", lw=1.0, ls="--")
            for col in range(row):
                axes[row, col].scatter(
                    truth_values[col],
                    truth_values[row],
                    marker="*",
                    color="black",
                    s=28,
                    zorder=10,
                )
                axes[row, col].scatter(
                    map_values[col],
                    map_values[row],
                    marker="x",
                    color="#E69F00",
                    s=22,
                    zorder=10,
                )
        figure.suptitle(
            f"{item.example_key} | row {int(item.row_index)} | truth star, MAP x",
            fontsize=11,
        )
        figure.savefig(galaxy_dir / f"{subset_name}.png", dpi=180)
        figure.savefig(galaxy_dir / f"{subset_name}.pdf")
        plt.close(figure)


def _write_sed_and_photometry(galaxy_dir: Path, runtime: Runtime, item) -> None:
    import matplotlib.pyplot as plt

    names = runtime.latent_spec.names
    truth = pd.read_parquet(galaxy_dir / "truth.parquet")
    map_frame = pd.read_parquet(galaxy_dir / "map_solutions.parquet").sort_values(
        "objective"
    )
    sources = {
        "Truth": truth[list(names)].to_numpy(dtype=float),
        "MAP": map_frame.iloc[:1][list(names)].to_numpy(dtype=float),
        "Encoder": _even_subsample(
            pd.read_parquet(galaxy_dir / "encoder_samples.parquet")[list(names)],
            32,
        ),
        "Encoder + IS": _even_subsample(
            pd.read_parquet(galaxy_dir / "importance_resampled_samples.parquet")[
                list(names)
            ],
            32,
        ),
        "NUTS": _even_subsample(
            pd.read_parquet(galaxy_dir / "nuts" / "samples.parquet")[list(names)],
            32,
        ),
        "MCLMC": _even_subsample(
            pd.read_parquet(galaxy_dir / "mclmc" / "samples.parquet")[list(names)],
            32,
        ),
    }
    colors = {
        "Truth": "black",
        "MAP": "#E69F00",
        "Encoder": "#0072B2",
        "Encoder + IS": "#009E73",
        "NUTS": "#D55E00",
        "MCLMC": "#CC79A7",
    }
    sed_artifacts = {}
    prediction_rows = []
    for label, theta in sources.items():
        result = predict_batch_seds(runtime.context, list(names), theta)
        flux = 10.0 ** (-0.4 * (result.model_mags + 48.6))
        sed_artifacts[label] = (result, flux)
        for band_index, band in enumerate(runtime.arrays.band_names):
            prediction_rows.append(
                {
                    "method": label,
                    "band": band,
                    "flux_q16": np.quantile(flux[:, band_index], 0.16),
                    "flux_q50": np.quantile(flux[:, band_index], 0.50),
                    "flux_q84": np.quantile(flux[:, band_index], 0.84),
                }
            )
    _write_parquet(
        pd.DataFrame(prediction_rows),
        galaxy_dir / "photometric_predictions.parquet",
    )
    np.savez_compressed(
        galaxy_dir / "sed_draws.npz",
        wave=next(iter(sed_artifacts.values()))[0].wave,
        **{
            f"{label.lower().replace(' ', '_').replace('+', 'plus')}_dusted_rest_sed": value[0].dusted_rest_sed
            for label, value in sed_artifacts.items()
        },
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.5, 10.0),
        gridspec_kw={"height_ratios": [2.0, 1.35, 0.8]},
    )
    ax_sed, ax_flux, ax_residual = axes
    for label, (result, _flux) in sed_artifacts.items():
        wave = result.wave
        sed = result.dusted_rest_sed
        q16, q50, q84 = np.quantile(sed, [0.16, 0.5, 0.84], axis=0)
        valid = (wave >= 800.0) & (wave <= 30_000.0) & np.isfinite(q50) & (q50 > 0)
        ax_sed.plot(wave[valid], q50[valid], color=colors[label], lw=1.3, label=label)
        if len(sed) > 1:
            ax_sed.fill_between(
                wave[valid], q16[valid], q84[valid], color=colors[label], alpha=0.12
            )
    ax_sed.set_xscale("log")
    ax_sed.set_yscale("log")
    ax_sed.set_ylabel("Dusted rest SED [Lsun Hz$^{-1}$]")
    ax_sed.set_title(
        f"{item.example_key}: DSPS SED evaluated at truth parameters and posterior fits"
    )
    ax_sed.legend(ncol=3, fontsize=8)
    wavelengths = np.asarray(
        [
            np.sum(curve.wave * curve.transmission)
            / np.maximum(np.sum(curve.transmission), 1.0e-30)
            for curve in runtime.filters.values()
        ]
    )
    observation = runtime.arrays.flux[0]
    error = runtime.arrays.flux_err[0]
    ax_flux.errorbar(
        wavelengths,
        observation,
        yerr=error,
        fmt="o",
        color="black",
        capsize=2,
        label="Observed photometry",
    )
    for label, (_result, flux) in sed_artifacts.items():
        q16, q50, q84 = np.quantile(flux, [0.16, 0.5, 0.84], axis=0)
        ax_flux.plot(wavelengths, q50, marker=".", color=colors[label], label=label)
        if len(flux) > 1:
            ax_flux.fill_between(wavelengths, q16, q84, color=colors[label], alpha=0.12)
        residual = (q50 - observation) / error
        ax_residual.plot(
            wavelengths, residual, marker=".", color=colors[label], label=label
        )
    for index, (_band, curve) in enumerate(runtime.filters.items()):
        transmission = curve.transmission / np.maximum(
            np.max(curve.transmission), 1.0e-30
        )
        ax_flux.fill_between(
            curve.wave,
            0.0,
            transmission * np.nanmax(observation) * 0.18,
            alpha=0.08,
            color=plt.cm.tab20(index % 20),
        )
    ax_flux.set_xscale("log")
    ax_flux.set_ylabel("Observed $f_\\nu$ [cgs]")
    ax_flux.legend(ncol=3, fontsize=7)
    ax_residual.axhline(0.0, color="black", lw=0.8)
    ax_residual.axhspan(-1.0, 1.0, color="0.5", alpha=0.12)
    ax_residual.set_xscale("log")
    ax_residual.set_xlabel("Observed wavelength [Angstrom]")
    ax_residual.set_ylabel("(model - obs) / flux_err")
    fig.tight_layout()
    fig.savefig(galaxy_dir / "sed_photometry_comparison.png", dpi=190)
    fig.savefig(galaxy_dir / "sed_photometry_comparison.pdf")
    plt.close(fig)


def _write_convergence_plots(galaxy_dir: Path, names: tuple[str, ...]) -> None:
    import matplotlib.pyplot as plt

    for sampler in ("nuts", "mclmc"):
        frame = pd.read_parquet(galaxy_dir / sampler / "samples.parquet")
        shown = names[:5]
        fig, axes = plt.subplots(len(shown), 1, figsize=(10, 1.8 * len(shown)), sharex=True)
        for ax, name in zip(np.atleast_1d(axes), shown, strict=True):
            for chain, group in frame.groupby("chain"):
                ax.plot(group["draw"], group[name], lw=0.55, alpha=0.75, label=f"c{chain}")
            ax.set_ylabel(name)
        axes[0].legend(ncol=4, fontsize=7)
        axes[-1].set_xlabel("stored draw")
        fig.suptitle(f"{sampler.upper()} chain traces")
        fig.tight_layout()
        fig.savefig(galaxy_dir / f"{sampler}_trace.png", dpi=170)
        plt.close(fig)
        diagnostics = pd.read_parquet(galaxy_dir / sampler / "diagnostics.parquet")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        axes[0].barh(diagnostics["parameter"], diagnostics["rhat"])
        axes[0].axvline(1.01, color="red", ls="--")
        axes[0].set_title("split R-hat")
        axes[1].barh(diagnostics["parameter"], diagnostics["bulk_ess"])
        axes[1].axvline(400, color="red", ls="--")
        axes[1].set_title("bulk ESS")
        axes[2].barh(diagnostics["parameter"], diagnostics["tail_ess"])
        axes[2].axvline(400, color="red", ls="--")
        axes[2].set_title("tail ESS")
        fig.tight_layout()
        fig.savefig(galaxy_dir / f"{sampler}_convergence.png", dpi=170)
        plt.close(fig)


def _cohort_item(out: Path, galaxy_index: int):
    cohort = pd.read_parquet(out / "cohort.parquet")
    if galaxy_index < 0 or galaxy_index >= len(cohort):
        raise IndexError(f"galaxy-index must be in [0, {len(cohort) - 1}]")
    return next(cohort.iloc[[galaxy_index]].itertuples(index=False))


def _galaxy_dir(out: Path, item) -> Path:
    return out / "galaxies" / f"{int(item.order):02d}_{item.example_key}_row{int(item.row_index)}"


def _sample_chunks(args: argparse.Namespace) -> tuple[int, ...]:
    if args.sample_chunks:
        chunks = tuple(int(value) for value in args.sample_chunks.split(","))
    elif args.mode == "smoke":
        chunks = (10,)
    elif args.mode == "pilot":
        chunks = (100, 500)
    else:
        chunks = (100, 500, 1000)
    if any(value <= 0 for value in chunks):
        raise ValueError("sample chunks must be positive")
    return chunks


def _even_subsample(frame: pd.DataFrame, size: int) -> np.ndarray:
    values = frame.to_numpy(dtype=float)
    if len(values) <= size:
        return values
    return values[np.linspace(0, len(values) - 1, size).astype(int)]


def _required_index(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"--{name} is required")
    return int(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


if __name__ == "__main__":
    main()
