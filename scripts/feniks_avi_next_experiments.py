"""Prepare and run the expert-capacity and population-prior AVI follow-up."""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.amortized.avi_experiments import Arm
from scripts.feniks_avi_experiments import read, sha, write

NEXT_ARMS = (
    Arm("H2_experts", experts=2),
    Arm("H8_experts", experts=8),
    Arm(
        "P_latest_prior",
        experts=4,
        kind="prior",
        prior_initialization="learned_source",
    ),
    Arm(
        "P_scratch_prior",
        experts=4,
        kind="prior",
        prior_initialization="identity_standard_normal",
    ),
)


def prepare(training: Path, root: Path) -> None:
    from scripts.feniks_avi_experiments import check_inputs

    if root.exists():
        raise FileExistsError(root)
    upstream = check_inputs(training)
    b = training / "arms/B_experts"
    b_final = read(b / "FINAL.json")
    if b_final.get("status") != "TRAINING_COMPLETE":
        raise ValueError("completed B_experts source required")
    required = (
        "source_config.yaml",
        "train.npy",
        "validation.npy",
        "teachers.npz",
        "teachers.json",
    )
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    for name in required:
        shutil.copy2(training / name, root / name)
    hashes = {
        str((training / "MANIFEST.json").resolve()): sha(training / "MANIFEST.json"),
        str((b / "encoder.eqx").resolve()): sha(b / "encoder.eqx"),
        str((b / "FINAL.json").resolve()): sha(b / "FINAL.json"),
    }
    hashes.update({str((root / name).resolve()): sha(root / name) for name in required})
    for folder in ("euclid_dsps", "scripts", "configs"):
        for path in Path(folder).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".yaml", ".slurm", ".sh"):
                hashes[str(path.resolve())] = sha(path)
    manifest = {
        **{
            key: upstream[key]
            for key in (
                "source",
                "seed",
                "train_rows",
                "validation_rows",
                "validation_catalog",
                "global_batch",
                "local_microbatch",
                "accumulation",
                "gpus",
                "particles",
                "decoder_draw_block",
                "validation_particles",
                "learning_rate",
                "warmup_fraction",
                "bootstrap_epochs",
                "cycle",
            )
        },
        "version": 1,
        "suite": "expert_capacity_and_selection_corrected_prior_v1",
        "arms": [asdict(arm) for arm in NEXT_ARMS],
        "epochs": 60,
        "hashes": hashes,
        "upstream_training": str(training.resolve()),
        "upstream_b_encoder": str((b / "encoder.eqx").resolve()),
        "prior_sweeps": 5,
        "prior_macro_objects": 1024,
        "prior_particles": 256,
        "prior_learning_rate": 1.0e-5,
        "prior_trust_strength": 0.2,
        "prior_maximum_kl_per_dimension": 0.02,
        "selection_correction": "required +log_alpha_eta in prior objective",
        "selection_in_object_weights": False,
        "posterior_weight_contract": "exact full_15d logtarget-logproposal",
        "primary_scientific_metrics": "physical_5d MIRA plus 5D population closure",
        "sfh_role": "retained 10D nuisance; never removed from target or weights",
        "truth_used": False,
        "scientific_promotion": False,
    }
    write(root / "MANIFEST.json", manifest)
    print(
        f"Prepared {root}: 2/8-expert capacity and latest/scratch prior array",
        flush=True,
    )


def _check(root: Path) -> dict:
    manifest = read(root / "MANIFEST.json")
    for path, digest in manifest["hashes"].items():
        if sha(path) != digest:
            raise ValueError(f"changed input: {path}")
    return manifest


def _prior_bank(model, candidate, runtime, particles: int, devices):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.avi_experiments import (
        normalized_weights,
        stratified_proposal,
    )
    from euclid_dsps.amortized.posterior_target import posterior_log_target

    @partial(eqx.filter_pmap, in_axes=(None, None, 0, 0), devices=devices)
    def evaluate(active_model, encoder, batch, key):
        x, logr = stratified_proposal(
            active_model, encoder, batch.features, key, particles
        )

        def decode(block):
            return posterior_log_target(
                active_model,
                block,
                batch,
                runtime.latent_spec,
                runtime.context,
                runtime.model_args,
                runtime.parameter_names,
                runtime.likelihood_config,
                runtime.calibration_config,
            )

        block = 8
        values = jax.lax.map(decode, x.reshape((-1, block) + x.shape[1:]))
        target = values.logtarget.reshape(x.shape[:-1])
        model_flux = values.model_flux.reshape(
            (x.shape[0],) + values.model_flux.shape[2:]
        )
        weights, valid, ess = normalized_weights(target - logr)
        residual = jnp.where(
            batch.mask[None],
            (model_flux - batch.flux[None]) / jnp.maximum(batch.flux_err[None], 1e-30),
            0.0,
        )
        squared = jnp.sum(residual**2, axis=-1) / jnp.maximum(
            jnp.sum(batch.mask, axis=-1), 1
        )
        return (
            x,
            weights,
            jnp.stack(
                (
                    ess,
                    jnp.max(weights, axis=0),
                    valid,
                    jnp.sqrt(jnp.mean(squared, axis=0)),
                    jnp.sqrt(jnp.sum(weights * squared, axis=0)),
                ),
                axis=-1,
            ),
        )

    return evaluate


def run_prior(root: Path, task: int, *, preflight: bool, platform: str = "gpu") -> None:
    import fcntl

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    from euclid_dsps.amortized.adaptive_smc_trainer import (
        _make_selection_log_alpha_fn,
        prepare_adaptive_training_runtime,
    )
    from euclid_dsps.amortized.features import make_encoder_features
    from euclid_dsps.amortized.sc_drws_trainer import _apply_prior_updates
    from euclid_dsps.amortized.train import (
        LossBatch,
        build_prior_from_config,
        load_checkpoint,
    )
    from euclid_dsps.config import load_config
    from scripts.feniks_avi_experiments import (
        initialize_candidate,
        initialize_transport,
    )

    manifest = _check(root)
    arm = Arm(**manifest["arms"][task])
    if arm.kind != "prior":
        raise ValueError("prior runner received an encoder arm")
    out = root / ("preflight" if preflight else "arms") / arm.name
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "FINAL.json").exists():
        print(f"Already complete: {out}", flush=True)
        return
    if not preflight:
        receipt = read(root / "preflight" / arm.name / "FINAL.json")
        if receipt.get("status") != "PREFLIGHT_PASS":
            raise ValueError("matching prior preflight required")
    devices = tuple(jax.local_devices())
    if len(devices) != 4 or any(device.platform != platform for device in devices):
        raise ValueError("exactly four GPU devices required")
    if not jax.config.x64_enabled:
        raise ValueError("JAX_ENABLE_X64=true required")
    started = time.monotonic()
    write(out / "PROGRESS.json", {"stage": "loading", "arm": arm.name})
    config = load_config(root / "source_config.yaml")
    model = load_checkpoint(manifest["source"]["checkpoint"], config)
    config = copy.deepcopy(config)
    config["amortized"]["encoder"]["transport_float64"] = True
    model = eqx.tree_at(
        lambda item: item.encoder, model, initialize_transport(model.encoder)
    )
    runtime = prepare_adaptive_training_runtime(
        config,
        out / "runtime",
        train_indices_file=root / "train.npy",
        validation_indices_file=root / "validation.npy",
        validation_catalog_path=manifest["validation_catalog"],
        fixed_feature_stats_path=manifest["source"]["feature_stats"],
        train_population_prior=True,
    )
    b_arm = Arm("B_experts", experts=4)
    template = initialize_candidate(
        model, config, runtime.latent_spec, b_arm, manifest["seed"]
    )
    candidate = eqx.tree_deserialise_leaves(manifest["upstream_b_encoder"], template)
    if arm.prior_initialization == "identity_standard_normal":
        fresh = build_prior_from_config(
            config,
            jax.random.PRNGKey(manifest["seed"] + 91000000),
            latent_dim=len(runtime.latent_spec.names),
            active_spec=runtime.latent_spec,
        )
        model = eqx.tree_at(lambda item: item.prior, model, fresh)
    initial_prior = jax.tree_util.tree_map(lambda value: value, model.prior)
    optimizer = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adam(manifest["prior_learning_rate"]),
    )
    optimizer_state = optimizer.init(eqx.filter(model.prior, eqx.is_inexact_array))
    selection_fn = _make_selection_log_alpha_fn(runtime)

    arrays = runtime.train_arrays

    def make_batch(indices):
        flux = np.asarray(arrays.flux)[indices]
        error = np.asarray(arrays.flux_err)[indices]
        mask = np.asarray(arrays.mask)[indices]
        features = make_encoder_features(flux, error, runtime.feature_stats, mask)
        return LossBatch(
            jnp.asarray(flux),
            jnp.asarray(error),
            jnp.asarray(mask),
            features,
            jnp.zeros((len(indices), 0), jnp.float32),
        )

    particle_count = 64 if preflight else int(manifest["prior_particles"])
    bank = _prior_bank(model, candidate, runtime, particle_count, devices)
    macro_objects = 256 if preflight else int(manifest["prior_macro_objects"])
    sweeps = 1 if preflight else int(manifest["prior_sweeps"])
    macros_per_sweep = math.ceil(len(arrays.flux) / macro_objects)
    total = 1 if preflight else sweeps * macros_per_sweep
    first = 0
    if not preflight and (out / "RESUME.json").exists():
        resume = read(out / "RESUME.json")
        if (
            resume["manifest_sha256"] != sha(root / "MANIFEST.json")
            or sha(out / resume["path"]) != resume["sha256"]
        ):
            raise ValueError("prior resume contract mismatch")
        model_prior, optimizer_state = eqx.tree_deserialise_leaves(
            out / resume["path"], (model.prior, optimizer_state)
        )
        model = eqx.tree_at(lambda item: item.prior, model, model_prior)
        first = int(resume["next_macro"])
    history = out / "prior_training.csv"
    last_rows = None
    for absolute in range(first, total):
        sweep, macro = divmod(absolute, macros_per_sweep)
        order = np.random.default_rng(manifest["seed"] + sweep).permutation(
            len(arrays.flux)
        )
        order = np.resize(order, macros_per_sweep * macro_objects)
        indices = order[macro * macro_objects : (macro + 1) * macro_objects]
        particle_parts, weight_parts, metric_parts = [], [], []
        for offset in range(0, macro_objects, 256):
            local = indices[offset : offset + 256]
            payload = jax.tree_util.tree_map(
                lambda value: value.reshape((4, 64) + value.shape[1:]),
                make_batch(local),
            )
            keys = jax.random.split(
                jax.random.fold_in(
                    jax.random.PRNGKey(manifest["seed"] + 92000000),
                    absolute * 8 + offset,
                ),
                4,
            )
            particles, weights, metrics = jax.device_get(
                bank(model, candidate, payload, keys)
            )
            particle_parts.append(
                np.asarray(particles)
                .transpose(1, 0, 2, 3)
                .reshape(particle_count, 256, -1)
            )
            weight_parts.append(
                np.asarray(weights).transpose(1, 0, 2).reshape(particle_count, 256)
            )
            metric_parts.append(np.asarray(metrics).reshape(256, 5))
        particles = np.concatenate(particle_parts, axis=1)
        weights = np.concatenate(weight_parts, axis=1)
        metrics = np.concatenate(metric_parts, axis=0)
        prior_cfg = {
            "minimum_finite_objects": 1,
            "minimum_median_ess_fraction": 0.0,
            "maximum_median_weight": 1.0,
            "maximum_unresolved_fraction": 1.0,
            "population_first": True,
            "population_stability_diagnostics": True,
            "updates_per_macro": 1,
            "trust_samples": 1024,
            "trust_strength": manifest["prior_trust_strength"],
            "maximum_kl_per_dimension": manifest["prior_maximum_kl_per_dimension"],
            "maximum_alpha_mc_relative_error": 0.15,
            "enforce_alpha_mc_relative_error": True,
            "enforce_hard_trust_region": True,
        }
        _gate, optimizer_state, model, _count, rows = _apply_prior_updates(
            model=model,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            selection_fn=selection_fn,
            particles=jnp.asarray(particles),
            weights=jnp.asarray(weights),
            ess_fraction=metrics[:, 0] / particle_count,
            max_weight=metrics[:, 1],
            finite=metrics[:, 2].astype(bool),
            unresolved=np.zeros(len(metrics), dtype=bool),
            prior_cfg=prior_cfg,
            key=jax.random.fold_in(
                jax.random.PRNGKey(manifest["seed"] + 93000000), absolute
            ),
            prior_updates=absolute,
            epoch=sweep,
            macro_index=macro,
        )
        last_rows = rows
        row = {
            **rows[-1],
            "absolute_macro": absolute + 1,
            "total_macros": total,
            "posterior_ess_median": float(np.median(metrics[:, 0])),
            "posterior_ess_q10": float(np.quantile(metrics[:, 0], 0.1)),
            "posterior_max_weight_q90": float(np.quantile(metrics[:, 1], 0.9)),
            "raw_predictive_rms_median": float(np.median(metrics[:, 3])),
            "is_predictive_rms_median": float(np.median(metrics[:, 4])),
            "elapsed_seconds": time.monotonic() - started,
        }
        pd.DataFrame([row]).to_csv(
            history, mode="a", header=not history.exists(), index=False
        )
        if not row["update_applied"]:
            raise RuntimeError(f"prior update rejected: {row}")
        write(
            out / "PROGRESS.json",
            {
                "stage": "prior_update_complete",
                "arm": arm.name,
                "sweep": sweep + 1,
                "macro": macro + 1,
                "macros_per_sweep": macros_per_sweep,
                "progress_percent": 100 * (absolute + 1) / total,
                **{
                    key: row[key]
                    for key in (
                        "posterior_ess_median",
                        "posterior_ess_q10",
                        "posterior_max_weight_q90",
                        "loss",
                        "log_alpha",
                        "proposed_kl",
                    )
                },
            },
        )
        print(
            f"[avi-prior] {arm.name} sweep={sweep + 1}/{sweeps} macro={macro + 1}/{macros_per_sweep} "
            f"ess={row['posterior_ess_median']:.2f} loss={row['loss']:.5g}",
            flush=True,
        )
        if not preflight:
            state_path = out / f"prior_state_{absolute + 1:05d}.eqx"
            eqx.tree_serialise_leaves(state_path, (model.prior, optimizer_state))
            write(
                out / "RESUME.json",
                {
                    "next_macro": absolute + 1,
                    "path": state_path.name,
                    "sha256": sha(state_path),
                    "manifest_sha256": sha(root / "MANIFEST.json"),
                },
            )
            old = sorted(out.glob("prior_state_*.eqx"))
            for path in old[:-2]:
                path.unlink()
    if last_rows is None:
        raise RuntimeError("prior experiment executed no macro update")
    changed = any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree_util.tree_leaves(eqx.filter(initial_prior, eqx.is_array)),
            jax.tree_util.tree_leaves(eqx.filter(model.prior, eqx.is_array)),
            strict=True,
        )
    )
    if not changed:
        raise RuntimeError("prior optimizer applied no parameter change")
    if not preflight:
        eqx.tree_serialise_leaves(out / "prior.eqx", model.prior)
    write(
        out / "FINAL.json",
        {
            "status": "PREFLIGHT_PASS" if preflight else "PRIOR_TRAINING_COMPLETE",
            "arm": asdict(arm),
            "manifest_sha256": sha(root / "MANIFEST.json"),
            "prior_initialization": arm.prior_initialization,
            "prior_updates": total,
            "encoder": "frozen B_experts",
            "selection_correction": "+log_alpha_eta differentiated at every update",
            "selection_in_object_weights": False,
            "full_15d_target": True,
            "physical_5d_primary_evaluation": True,
            "truth_used": False,
            "elapsed_seconds": time.monotonic() - started,
            "prior_sha256": sha(out / "prior.eqx") if not preflight else None,
            "scientific_promotion": False,
        },
    )


def run(args) -> None:
    manifest = _check(args.root)
    arm = Arm(**manifest["arms"][args.task])
    if arm.kind == "encoder":
        from scripts.feniks_avi_experiments import run as run_encoder

        run_encoder(args)
    else:
        run_prior(args.root, args.task, preflight=args.mode == "preflight")


def summarize(root: Path) -> None:
    manifest = _check(root)
    rows = []
    upstream = Path(manifest["upstream_training"])
    for arm, directory in (
        ("B_experts", upstream / "arms/B_experts"),
        ("H2_experts", root / "arms/H2_experts"),
        ("H8_experts", root / "arms/H8_experts"),
    ):
        frame = pd.read_csv(directory / "validation_final.csv")
        values = frame.groupby("validation_index").median(numeric_only=True)
        record = {
            "arm": arm,
            "kind": "encoder_capacity",
            "median_ess_fraction": float(values.ess_fraction.median()),
            "median_max_weight": float(values.max_weight.median()),
            "fraction_ess_below_5": float((values.ess < 5).mean()),
            "raw_predictive_rms": float(values.raw_predictive_rms.median()),
        }
        audit = root / "expert_audits" / arm / "EXPERT_AUDIT.json"
        if audit.exists():
            record["physical_5d_mira_raw"] = read(audit)["physical_5d_mira_raw"]
        rows.append(record)
    for arm in ("P_latest_prior", "P_scratch_prior"):
        frame = pd.read_csv(root / "arms" / arm / "prior_training.csv")
        last = frame.iloc[-1]
        rows.append(
            {
                "arm": arm,
                "kind": "prior_only",
                "median_ess_fraction": float(
                    last.posterior_ess_median / manifest["prior_particles"]
                ),
                "median_max_weight": float(last.posterior_max_weight_q90),
                "raw_predictive_rms": float(last.raw_predictive_rms_median),
                "is_predictive_rms": float(last.is_predictive_rms_median),
                "selection_log_alpha": float(last.log_alpha),
                "proposed_kl": float(last.proposed_kl),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(root / "next_summary.csv", index=False)
    print(result.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("prepare", "preflight", "train", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=18.0)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.training is None:
            parser.error("prepare requires --training")
        prepare(args.training, args.root)
    elif args.mode == "summarize":
        summarize(args.root)
    else:
        run(args)


if __name__ == "__main__":
    main()
