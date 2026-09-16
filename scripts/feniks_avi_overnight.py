"""Overnight B4/prior coadaptation, exact IS inference and closure reports."""

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
from euclid_dsps.amortized.mira import FENIKS_SPLINE15D_PARAMETERS
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_avi_inference import joint_resample, sample_frame

REFRESH_ARMS = (
    Arm("Q_source_refresh", experts=4),
    Arm("Q_latest_refresh", experts=4),
)


def _files_hash(paths) -> dict[str, str]:
    return {str(Path(path).resolve()): sha(path) for path in paths}


def _check_declared_hashes(manifest: dict) -> None:
    for path, digest in manifest.get("hashes", {}).items():
        if sha(path) != digest:
            raise ValueError(f"changed declared input: {path}")


def _require_final(path: Path, status: str, manifest_path: Path) -> dict:
    final = read(path)
    if final.get("status") != status:
        raise ValueError(f"expected {status}: {path}")
    expected = final.get("manifest_sha256")
    if expected is not None and expected != sha(manifest_path):
        raise ValueError(f"stale final receipt: {path}")
    for field, filename in (
        ("encoder_sha256", "encoder.eqx"),
        ("prior_sha256", "prior.eqx"),
    ):
        digest = final.get(field)
        artifact = path.parent / filename
        if digest is not None and sha(artifact) != digest:
            raise ValueError(f"changed final artifact: {artifact}")
    return final


def prepare_training(source: Path, prior_followup: Path, root: Path) -> None:
    """Prepare two duration-matched q refreshes without reading truth."""
    from scripts.feniks_avi_experiments import check_inputs

    if root.exists():
        raise FileExistsError(root)
    source_manifest = check_inputs(source)
    prior_manifest = read(prior_followup / "MANIFEST.json")
    _check_declared_hashes(prior_manifest)
    b_dir = source / "arms/B_experts"
    _require_final(b_dir / "FINAL.json", "TRAINING_COMPLETE", source / "MANIFEST.json")
    latest_dir = prior_followup / "arms/P_latest_prior"
    scratch_dir = prior_followup / "arms/P_scratch_prior"
    for directory in (latest_dir, scratch_dir):
        _require_final(
            directory / "FINAL.json",
            "PRIOR_TRAINING_COMPLETE",
            prior_followup / "MANIFEST.json",
        )
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
        shutil.copy2(source / name, root / name)
    paths = [
        source / "MANIFEST.json",
        b_dir / "encoder.eqx",
        b_dir / "FINAL.json",
        prior_followup / "MANIFEST.json",
        latest_dir / "prior.eqx",
        latest_dir / "FINAL.json",
        scratch_dir / "prior.eqx",
        scratch_dir / "FINAL.json",
        *(root / name for name in required),
    ]
    manifest = {
        **{
            key: source_manifest[key]
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
                "cycle",
            )
        },
        "version": 1,
        "suite": "b4_prior_coadaptation_q_refresh_v1",
        "arms": [asdict(arm) for arm in REFRESH_ARMS],
        "epochs": 12,
        "bootstrap_epochs": 0,
        "initial_encoder_checkpoint_by_arm": {
            arm.name: str((b_dir / "encoder.eqx").resolve()) for arm in REFRESH_ARMS
        },
        "initial_prior_checkpoint_by_arm": {
            "Q_source_refresh": None,
            "Q_latest_refresh": str((latest_dir / "prior.eqx").resolve()),
        },
        "control": "same B4 initialization and duration; only frozen prior differs",
        "decoder_frozen": True,
        "prior_frozen_during_q_refresh": True,
        "selection_correction": (
            "selection-aware sleep plus frozen log_alpha receipt; log_alpha is "
            "constant with respect to the q-only wake objective"
        ),
        "selection_in_object_weights": False,
        "truth_used": False,
        "scientific_promotion": False,
        "source_training": str(source.resolve()),
        "prior_followup": str(prior_followup.resolve()),
        "p_scratch_role": "inference-only negative control",
        "hashes": _files_hash(paths),
        "prior_followup_suite": prior_manifest.get("suite"),
    }
    write(root / "MANIFEST.json", manifest)
    print(f"Prepared q refresh root: {root}", flush=True)


def _parent_catalog(selected_catalog: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    if selected_catalog.parent.name != "amortized":
        raise ValueError("cannot infer parent C0 catalog; pass --parent-catalog")
    return selected_catalog.parent.parent / selected_catalog.name


def _truth_table(path: Path, rows: np.ndarray | None = None) -> pd.DataFrame:
    names = list(FENIKS_SPLINE15D_PARAMETERS)
    frame = pd.read_parquet(path, columns=names)
    if rows is not None:
        frame = frame.iloc[rows].copy()
        identities = np.asarray(rows, dtype=np.int64)
    else:
        frame = frame.copy()
        identities = np.arange(len(frame), dtype=np.int64)
    values = frame[names].to_numpy(dtype=np.float64)
    if not len(frame) or not np.isfinite(values).all():
        raise ValueError(f"finite canonical 15D truth required: {path}")
    frame.insert(0, "row_index", identities)
    frame.insert(0, "object_id", identities)
    return frame


def prepare_inference(
    training: Path,
    source: Path,
    prior_followup: Path,
    root: Path,
    *,
    parent_catalog: Path | None,
    particles: int,
) -> None:
    """Freeze five encoder/prior pairs and truth cohorts after q training."""
    if root.exists():
        raise FileExistsError(root)
    if particles < 256 or particles % 8:
        raise ValueError("particles must be >=256 and divisible by eight")
    training_manifest = read(training / "MANIFEST.json")
    _check_declared_hashes(training_manifest)
    for arm in REFRESH_ARMS:
        _require_final(
            training / "arms" / arm.name / "FINAL.json",
            "TRAINING_COMPLETE",
            training / "MANIFEST.json",
        )
    source_manifest = read(source / "MANIFEST.json")
    _check_declared_hashes(source_manifest)
    _require_final(
        source / "arms/B_experts/FINAL.json",
        "TRAINING_COMPLETE",
        source / "MANIFEST.json",
    )
    prior_manifest = read(prior_followup / "MANIFEST.json")
    _check_declared_hashes(prior_manifest)
    for name in ("P_latest_prior", "P_scratch_prior"):
        _require_final(
            prior_followup / "arms" / name / "FINAL.json",
            "PRIOR_TRAINING_COMPLETE",
            prior_followup / "MANIFEST.json",
        )
    validation_rows = np.load(source / "validation.npy", allow_pickle=False)
    selected_catalog = Path(source_manifest["validation_catalog"]).resolve()
    parent_catalog = _parent_catalog(selected_catalog, parent_catalog)
    if not selected_catalog.is_file() or not parent_catalog.is_file():
        raise FileNotFoundError(
            f"selected/parent truth catalogs required: {selected_catalog}, {parent_catalog}"
        )
    b_encoder = source / "arms/B_experts/encoder.eqx"
    latest_prior = prior_followup / "arms/P_latest_prior/prior.eqx"
    scratch_prior = prior_followup / "arms/P_scratch_prior/prior.eqx"
    dependency_receipts = [
        source / "arms/B_experts/FINAL.json",
        training / "arms/Q_source_refresh/FINAL.json",
        training / "arms/Q_latest_refresh/FINAL.json",
        prior_followup / "arms/P_latest_prior/FINAL.json",
        prior_followup / "arms/P_scratch_prior/FINAL.json",
    ]
    variants = [
        {
            "name": "B_source",
            "encoder": str(b_encoder.resolve()),
            "prior": None,
            "prior_label": "source",
            "emit_prior_population": True,
        },
        {
            "name": "Q_source_refresh",
            "encoder": str((training / "arms/Q_source_refresh/encoder.eqx").resolve()),
            "prior": None,
            "prior_label": "source",
            "emit_prior_population": False,
        },
        {
            "name": "B_latest",
            "encoder": str(b_encoder.resolve()),
            "prior": str(latest_prior.resolve()),
            "prior_label": "latest",
            "emit_prior_population": True,
        },
        {
            "name": "Q_latest_refresh",
            "encoder": str((training / "arms/Q_latest_refresh/encoder.eqx").resolve()),
            "prior": str(latest_prior.resolve()),
            "prior_label": "latest",
            "emit_prior_population": False,
        },
        {
            "name": "B_scratch_prior",
            "encoder": str(b_encoder.resolve()),
            "prior": str(scratch_prior.resolve()),
            "prior_label": "scratch",
            "emit_prior_population": True,
        },
    ]
    inputs = [
        training / "MANIFEST.json",
        source / "MANIFEST.json",
        source / "source_config.yaml",
        source / "train.npy",
        source / "validation.npy",
        source / "teachers.npz",
        b_encoder,
        latest_prior,
        scratch_prior,
        *dependency_receipts,
        selected_catalog,
        parent_catalog,
    ]
    for variant in variants:
        inputs.append(Path(variant["encoder"]))
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    inference_truth = _truth_table(selected_catalog, validation_rows)
    selected_truth = _truth_table(selected_catalog)
    parent_truth = _truth_table(parent_catalog)
    inference_truth.to_parquet(root / "inference_truth.parquet", index=False)
    selected_truth.to_parquet(root / "selected_population_truth.parquet", index=False)
    parent_truth.to_parquet(root / "parent_population_truth.parquet", index=False)
    truth_paths = [
        root / "inference_truth.parquet",
        root / "selected_population_truth.parquet",
        root / "parent_population_truth.parquet",
    ]
    write(
        root / "MANIFEST.json",
        {
            "version": 1,
            "suite": "avi_b4_prior_population_closure_v1",
            "training": str(training.resolve()),
            "source_training": str(source.resolve()),
            "prior_followup": str(prior_followup.resolve()),
            "source": source_manifest["source"],
            "config": str((source / "source_config.yaml").resolve()),
            "train_indices": str((source / "train.npy").resolve()),
            "validation_indices": str((source / "validation.npy").resolve()),
            "validation_catalog": str(selected_catalog),
            "parent_catalog": str(parent_catalog),
            "variants": variants,
            "particles": particles,
            "saved_draws": 512,
            "replicas": 2,
            "objects": len(validation_rows),
            "seed": 26091317,
            "prior_samples": 65536,
            "prior_selected_resamples": 65536,
            "prior_mira_objects": min(512, len(validation_rows), len(parent_truth)),
            "posterior_weight_contract": "ordinary full_15d logtarget-logproposal",
            "selected_aggregate": "equal object mixture of ordinary-IS joint banks",
            "parent_projection": "selected aggregate reweighted jointly by 1/beta(theta)",
            "truth_role": "post-training closure and plotting only",
            "truth_used_for_training_or_weights": False,
            "scientific_promotion": False,
            "hashes": {
                **_files_hash(inputs),
                **_files_hash(truth_paths),
            },
        },
    )
    print(f"Prepared inference root with {len(variants)} variants: {root}", flush=True)


def _check_manifest(root: Path) -> dict:
    manifest = read(root / "MANIFEST.json")
    for path, digest in manifest["hashes"].items():
        if sha(path) != digest:
            raise ValueError(f"changed input: {path}")
    return manifest


def infer(root: Path, task: int, *, platform: str = "gpu") -> None:
    """Generate complete ordinary-IS banks and beta values for one pair."""
    import fcntl

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.adaptive_smc_trainer import (
        prepare_adaptive_training_runtime,
    )
    from euclid_dsps.amortized.avi_experiments import log_prob, normalized_weights
    from euclid_dsps.amortized.features import make_encoder_features
    from euclid_dsps.amortized.latent import x_to_theta
    from euclid_dsps.amortized.posterior_target import posterior_log_target
    from euclid_dsps.amortized.proposal_expressivity import sample_independent_mixture
    from euclid_dsps.amortized.sc_asmc_report import _prior_report_arrays
    from euclid_dsps.amortized.train import (
        LossBatch,
        _selection_log_beta_from_prior_samples,
        load_checkpoint,
    )
    from euclid_dsps.config import load_config
    from scripts.feniks_avi_experiments import (
        initialize_candidate,
        initialize_transport,
        load_optional_component,
    )

    manifest = _check_manifest(root)
    variant = manifest["variants"][task]
    dest = root / "arms" / variant["name"]
    dest.mkdir(parents=True, exist_ok=True)
    lock = (dest / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (dest / "FINAL.json").exists():
        return
    devices = tuple(jax.local_devices())
    if (
        len(devices) != 4
        or any(device.platform != platform for device in devices)
        or not jax.config.x64_enabled
    ):
        raise ValueError("exactly four devices with float64 required")
    started = time.monotonic()
    write(dest / "PROGRESS.json", {"stage": "loading", "arm": variant["name"]})
    config = load_config(manifest["config"])
    model = load_checkpoint(manifest["source"]["checkpoint"], config)
    config = copy.deepcopy(config)
    config["amortized"]["encoder"]["transport_float64"] = True
    model = eqx.tree_at(
        lambda item: item.encoder, model, initialize_transport(model.encoder)
    )
    model = eqx.tree_at(
        lambda item: item.prior,
        model,
        load_optional_component(variant["prior"], model.prior),
    )
    runtime_dir = dest / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime = prepare_adaptive_training_runtime(
        config,
        runtime_dir,
        train_indices_file=manifest["train_indices"],
        validation_indices_file=manifest["validation_indices"],
        validation_catalog_path=manifest["validation_catalog"],
        fixed_feature_stats_path=manifest["source"]["feature_stats"],
        train_population_prior=False,
    )
    if runtime.likelihood_config.get("type", "gaussian") != "gaussian":
        raise ValueError("qualified Gaussian target required")
    if not runtime.selection_objective_config["selection_correction"]["enabled"]:
        raise ValueError("selection model required for beta diagnostics")
    template = initialize_candidate(
        model, config, runtime.latent_spec, Arm("B4", experts=4), manifest["seed"]
    )
    candidate = eqx.tree_deserialise_leaves(variant["encoder"], template)
    arrays = runtime.validation_arrays
    rows = np.load(manifest["validation_indices"], allow_pickle=False)
    if not np.array_equal(arrays.row_index, rows):
        raise ValueError("validation identity order changed")
    features = make_encoder_features(
        arrays.flux, arrays.flux_err, runtime.feature_stats, arrays.mask
    )
    batch = jax.device_get(
        LossBatch(
            jnp.asarray(arrays.flux),
            jnp.asarray(arrays.flux_err),
            jnp.asarray(arrays.mask),
            features,
            jnp.zeros((len(rows), 0), jnp.float32),
        )
    )
    particles = int(manifest["particles"])
    selection = runtime.selection_objective_config["selection_correction"]

    def beta_for_object(x):
        return _selection_log_beta_from_prior_samples(
            model,
            x,
            runtime.latent_spec,
            runtime.context,
            runtime.model_args,
            runtime.parameter_names,
            runtime.calibration_config,
            selection,
        )

    @partial(eqx.filter_pmap, in_axes=(None, 0, 0), devices=devices)
    def bank(active_candidate, local_batch, key):
        x = sample_independent_mixture(
            model, active_candidate, key, local_batch.features, particles
        ).x
        logq = log_prob(model, active_candidate, local_batch.features, x)

        def decode(block):
            result = posterior_log_target(
                model,
                block,
                local_batch,
                runtime.latent_spec,
                runtime.context,
                runtime.model_args,
                runtime.parameter_names,
                runtime.likelihood_config,
                runtime.calibration_config,
            )
            residual = jnp.where(
                local_batch.mask[None],
                (result.model_flux - local_batch.flux[None])
                / jnp.maximum(local_batch.flux_err[None], 1e-30),
                0.0,
            )
            squared = jnp.sum(residual**2, axis=-1) / jnp.maximum(
                jnp.sum(local_batch.mask, axis=-1), 1
            )
            return result.logtarget, squared

        target, squared = jax.lax.map(decode, x.reshape((-1, 8) + x.shape[1:]))
        target, squared = target.reshape(logq.shape), squared.reshape(logq.shape)
        weights, valid, ess = normalized_weights(target - logq)
        log_beta = jax.vmap(beta_for_object, in_axes=1, out_axes=1)(x)
        metrics = jnp.stack(
            (
                ess,
                jnp.max(weights, axis=0),
                jnp.sqrt(jnp.mean(squared, axis=0)),
                jnp.sqrt(jnp.sum(weights * squared, axis=0)),
                jax.scipy.special.logsumexp(target - logq, axis=0) - jnp.log(particles),
                valid,
            ),
            axis=-1,
        )
        return x_to_theta(x, runtime.latent_spec), weights, log_beta, metrics

    records: list[dict] = []
    bank_hashes: dict[str, str] = {}
    for replica in range(int(manifest["replicas"])):
        raw_frames, is_frames = [], []
        for offset in range(0, len(rows), 4):
            indices = np.minimum(np.arange(offset, offset + 4), len(rows) - 1)
            payload = jax.tree_util.tree_map(
                lambda value, idx=indices: jnp.asarray(value[idx]).reshape(
                    (4, 1) + value.shape[1:]
                ),
                batch,
            )
            keys = jnp.stack(
                [
                    jax.random.fold_in(
                        jax.random.PRNGKey(manifest["seed"] + replica), int(index)
                    )
                    for index in indices
                ]
            )
            theta, weights, log_beta, metrics = jax.device_get(
                bank(candidate, payload, keys)
            )
            count = min(4, len(rows) - offset)
            theta = np.asarray(theta[:count, :, 0])
            weights = np.asarray(weights[:count, :, 0])
            log_beta = np.asarray(log_beta[:count, :, 0])
            metrics = np.asarray(metrics[:count, 0])
            if (
                not metrics[:, -1].all()
                or not np.isfinite(metrics).all()
                or np.any(np.isnan(log_beta))
                or np.any(log_beta > 1e-10)
            ):
                raise ValueError("invalid IS/beta bank; no fallback is permitted")
            bank_frames = []
            selected = []
            for local in range(count):
                row_index = int(rows[offset + local])
                rng = np.random.default_rng(
                    np.random.SeedSequence([manifest["seed"], replica, row_index, 99])
                )
                draws, draw_ids = joint_resample(
                    theta[local],
                    weights[local],
                    int(manifest["saved_draws"]),
                    rng,
                )
                selected.append(draws)
                records.append(
                    {
                        "row_index": row_index,
                        "replica": replica,
                        "ess": float(metrics[local, 0]),
                        "ess_fraction": float(metrics[local, 0] / particles),
                        "max_weight": float(metrics[local, 1]),
                        "raw_predictive_rms": float(metrics[local, 2]),
                        "is_predictive_rms": float(metrics[local, 3]),
                        "log_evidence": float(metrics[local, 4]),
                        "resampled_unique": int(len(np.unique(draw_ids))),
                    }
                )
                frame = sample_frame(
                    theta[local : local + 1], [row_index], runtime.latent_spec.names
                )
                frame["weight"] = weights[local]
                frame["log_beta"] = log_beta[local]
                bank_frames.append(frame)
            bank_dir = dest / f"bank_{replica}"
            bank_dir.mkdir(exist_ok=True)
            part = bank_dir / f"part_{offset:05d}.parquet"
            pd.concat(bank_frames, ignore_index=True).to_parquet(part, index=False)
            bank_hashes[str(part.relative_to(dest))] = sha(part)
            raw_frames.append(
                sample_frame(
                    theta[:, : int(manifest["saved_draws"])],
                    rows[offset : offset + count],
                    runtime.latent_spec.names,
                )
            )
            is_frames.append(
                sample_frame(
                    np.stack(selected),
                    rows[offset : offset + count],
                    runtime.latent_spec.names,
                )
            )
            write(
                dest / "PROGRESS.json",
                {
                    "stage": "inference",
                    "arm": variant["name"],
                    "replica": replica,
                    "objects_complete": offset + count,
                    "objects": len(rows),
                    "progress_percent": 100
                    * (replica * len(rows) + offset + count)
                    / (int(manifest["replicas"]) * len(rows)),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
        for label, frames in (("raw", raw_frames), ("is", is_frames)):
            pd.concat(frames, ignore_index=True).to_parquet(
                dest / f"{label}_{replica}.parquet", index=False
            )
    pd.DataFrame(records).to_csv(dest / "metrics.csv", index=False)
    write(dest / "BANK_MANIFEST.json", bank_hashes)
    if variant["emit_prior_population"]:
        arrays, summary = _prior_report_arrays(
            model,
            runtime,
            key=jax.random.fold_in(jax.random.PRNGKey(manifest["seed"]), 700 + task),
            n_samples=int(manifest["prior_samples"]),
            selected_resamples=int(manifest["prior_selected_resamples"]),
            decoder_batch_size=256,
        )
        np.savez_compressed(dest / "prior_population.npz", **arrays)
        write(dest / "PRIOR_POPULATION.json", summary)
    artifacts = {
        path.name: sha(path)
        for path in dest.glob("*")
        if path.is_file() and path.name not in {"FINAL.json", ".lock"}
    }
    write(
        dest / "FINAL.json",
        {
            "status": "INFERENCE_COMPLETE",
            "manifest_sha256": sha(root / "MANIFEST.json"),
            "variant": variant,
            "objects": len(rows),
            "particles": particles,
            "replicas": manifest["replicas"],
            "elapsed_seconds": time.monotonic() - started,
            "artifacts": artifacts,
            "bank_manifest_sha256": sha(dest / "BANK_MANIFEST.json"),
            "truth_used_for_sampling_or_weighting": False,
            "scientific_promotion": False,
        },
    )


def _normalized_log_weights(log_weight: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weight, dtype=np.float64)
    if np.any(np.isposinf(values)):
        raise ValueError("infinite inverse-selection weight")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("no finite population weights")
    maximum = np.max(values[finite])
    weights = np.where(finite, np.exp(values - maximum), 0.0)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("invalid population-weight normalization")
    return weights / total


def population_weights(bank: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return selected and inverse-beta parent mixture weights."""
    weight = pd.to_numeric(bank["weight"], errors="coerce").to_numpy(np.float64)
    log_beta = pd.to_numeric(bank["log_beta"], errors="coerce").to_numpy(np.float64)
    if np.any(weight < 0) or np.any(np.isnan(log_beta)) or np.any(log_beta > 1e-10):
        raise ValueError("invalid ordinary-IS or beta values")
    object_sum = (
        pd.Series(weight, index=bank.index)
        .groupby(bank["row_index"])
        .transform("sum")
        .to_numpy(np.float64)
    )
    if not np.isfinite(object_sum).all() or np.any(object_sum <= 0):
        raise ValueError("every object must have positive finite ordinary-IS mass")
    selected = np.divide(
        weight,
        object_sum,
        out=np.zeros_like(weight),
        where=np.isfinite(object_sum) & (object_sum > 0),
    )
    selected /= selected.sum()
    log_weight = np.full_like(weight, -np.inf)
    positive = weight > 0
    log_weight[positive] = np.log(weight[positive])
    parent = _normalized_log_weights(log_weight - np.log(object_sum) - log_beta)
    diagnostics = {
        "selected_ess": float(1.0 / np.sum(selected**2)),
        "selected_max_weight": float(selected.max()),
        "parent_projection_ess": float(1.0 / np.sum(parent**2)),
        "parent_projection_max_weight": float(parent.max()),
        "log_beta_q01": float(np.quantile(log_beta[np.isfinite(log_beta)], 0.01)),
        "log_beta_median": float(np.median(log_beta[np.isfinite(log_beta)])),
        "zero_beta_fraction": float(np.mean(np.isneginf(log_beta))),
    }
    return selected, parent, diagnostics


def _weighted_draw(values, weights, count, seed):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(values), size=count, replace=True, p=weights)
    return np.asarray(values)[indices]


def _save_figure(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _corner(path: Path, series: dict[str, np.ndarray], truth, names) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dimensions = len(names)
    fig, axes = plt.subplots(dimensions, dimensions, figsize=(13, 13), squeeze=False)
    colors = plt.get_cmap("tab10").colors
    rng = np.random.default_rng(260913)
    truth = None if truth is None else np.asarray(truth)
    for row in range(dimensions):
        for column in range(dimensions):
            axis = axes[row, column]
            if row < column:
                axis.axis("off")
                continue
            if row == column:
                if truth is not None and truth.ndim == 2:
                    axis.hist(
                        truth[:, column],
                        bins=40,
                        density=True,
                        histtype="step",
                        color="black",
                        linewidth=1.4,
                        label="truth",
                    )
                for color, (label, values) in zip(colors, series.items(), strict=False):
                    axis.hist(
                        values[:, column],
                        bins=40,
                        density=True,
                        histtype="step",
                        linewidth=1.1,
                        color=color,
                        label=label,
                    )
                if truth is not None and truth.ndim == 1:
                    axis.axvline(truth[column], color="black", linewidth=1.2)
            else:
                if truth is not None and truth.ndim == 2:
                    take = rng.choice(len(truth), min(700, len(truth)), replace=False)
                    axis.scatter(
                        truth[take, column],
                        truth[take, row],
                        s=3,
                        alpha=0.10,
                        color="black",
                        label="truth",
                    )
                for color, (label, values) in zip(colors, series.items(), strict=False):
                    take = rng.choice(len(values), min(700, len(values)), replace=False)
                    axis.scatter(
                        values[take, column],
                        values[take, row],
                        s=3,
                        alpha=0.10,
                        color=color,
                        label=label,
                    )
                if truth is not None and truth.ndim == 1:
                    axis.axvline(truth[column], color="black", linewidth=0.8)
                    axis.axhline(truth[row], color="black", linewidth=0.8)
            if row == dimensions - 1:
                axis.set_xlabel(names[column], fontsize=7)
            if column == 0 and row:
                axis.set_ylabel(names[row], fontsize=7)
            axis.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=6)
    return _save_figure(fig, path)


def _marginals(path: Path, series: dict[str, np.ndarray], truth, names) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = math.ceil(len(names) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(13, 2.7 * rows), squeeze=False)
    truth = None if truth is None else np.asarray(truth)
    for index, name in enumerate(names):
        axis = axes.flat[index]
        if truth is not None and truth.ndim == 2:
            axis.hist(
                truth[:, index],
                bins=45,
                density=True,
                histtype="step",
                color="black",
                linewidth=1.4,
                label="truth",
            )
        for label, values in series.items():
            axis.hist(
                values[:, index], bins=45, density=True, histtype="step", label=label
            )
        if truth is not None and truth.ndim == 1:
            axis.axvline(truth[index], color="black", linewidth=1.2, label="truth")
        axis.set_title(name, fontsize=8)
    for axis in axes.flat[len(names) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=6)
    return _save_figure(fig, path)


def _representative_rows(truth: pd.DataFrame) -> list[int]:
    names = list(FENIKS_SPLINE15D_PARAMETERS[:5])
    values = truth[names].to_numpy(np.float64)
    median = np.median(values, axis=0)
    scale = np.maximum(
        np.quantile(values, 0.84, axis=0) - np.quantile(values, 0.16, axis=0), 1e-8
    )
    distance = np.sqrt(np.mean(((values - median) / scale) ** 2, axis=1))
    requests = [
        (distance, 0.0),
        (values[:, 0], np.quantile(values[:, 0], 0.03)),
        (values[:, 0], np.quantile(values[:, 0], 0.985)),
        (values[:, 1], np.quantile(values[:, 1], 0.03)),
        (values[:, 1], np.quantile(values[:, 1], 0.985)),
        (values[:, 2], np.quantile(values[:, 2], 0.985)),
        (values[:, 3], np.quantile(values[:, 3], 0.985)),
        (values[:, 4], np.quantile(values[:, 4], 0.03)),
    ]
    used: set[int] = set()
    result = []
    for vector, target in requests:
        for position in np.argsort(np.abs(vector - target)):
            if int(position) not in used:
                used.add(int(position))
                result.append(int(truth.iloc[int(position)].row_index))
                break
    return result


def _prior_mira_frame(pool, truth: pd.DataFrame, names, draws: int, seed: int):
    rng = np.random.default_rng(seed)
    count = len(truth)
    indices = rng.integers(len(pool), size=(count, draws))
    return sample_frame(np.asarray(pool)[indices], truth.row_index.to_numpy(), names)


def report(root: Path) -> None:
    """Write posterior and population closure after every inference completes."""
    from euclid_dsps.amortized.mira import evaluate_feniks_mira
    from euclid_dsps.amortized.population_projection import distribution_comparison
    from euclid_dsps.amortized.sc_asmc_closure_analysis import _joint_metric_row

    manifest = _check_manifest(root)
    for variant in manifest["variants"]:
        dest = root / "arms" / variant["name"]
        final = read(dest / "FINAL.json")
        if final.get("status") != "INFERENCE_COMPLETE":
            raise ValueError(f"incomplete inference: {variant['name']}")
        for name, digest in final["artifacts"].items():
            if sha(dest / name) != digest:
                raise ValueError(
                    f"changed inference artifact: {variant['name']}/{name}"
                )
        banks = read(dest / "BANK_MANIFEST.json")
        for relative, digest in banks.items():
            if sha(dest / relative) != digest:
                raise ValueError(f"changed bank shard: {variant['name']}/{relative}")
    output = root / "report"
    output.mkdir(exist_ok=True)
    names = list(FENIKS_SPLINE15D_PARAMETERS)
    physical = names[:5]
    inference_truth = pd.read_parquet(root / "inference_truth.parquet")
    selected_truth = pd.read_parquet(root / "selected_population_truth.parquet")
    parent_truth = pd.read_parquet(root / "parent_population_truth.parquet")

    for replica in range(int(manifest["replicas"])):
        specs = [
            (
                f"{variant['name']}_{kind}",
                root / "arms" / variant["name"] / f"{kind}_{replica}.parquet",
            )
            for variant in manifest["variants"]
            for kind in ("raw", "is")
        ]
        evaluate_feniks_mira(
            truth_path=root / "inference_truth.parquet",
            posterior_specs=specs,
            out_dir=output / f"posterior_mira_{replica}",
            samples_per_object=int(manifest["saved_draws"]),
            seed=int(manifest["seed"]) + replica,
            num_regions=100,
            num_bootstrap=1000,
        )
    posterior_mira = []
    for replica in range(int(manifest["replicas"])):
        frame = pd.read_csv(output / f"posterior_mira_{replica}/mira_scores.csv")
        frame.insert(0, "replica", replica)
        posterior_mira.append(frame)
    pd.concat(posterior_mira, ignore_index=True).to_csv(
        output / "posterior_mira_scores.csv", index=False
    )

    prior_pools: dict[str, dict[str, np.ndarray]] = {}
    for variant in manifest["variants"]:
        if variant["emit_prior_population"]:
            with np.load(
                root / "arms" / variant["name"] / "prior_population.npz",
                allow_pickle=False,
            ) as archive:
                prior_pools[variant["prior_label"]] = {
                    "parent": np.asarray(archive["theta"]),
                    "selected": np.asarray(archive["selected_theta"]),
                }
    prior_mira_objects = int(manifest["prior_mira_objects"])
    selected_mira_truth = inference_truth.iloc[:prior_mira_objects].copy()
    selected_mira_truth.to_parquet(
        output / "selected_prior_mira_truth.parquet", index=False
    )
    parent_positions = np.unique(
        np.linspace(0, len(parent_truth) - 1, prior_mira_objects, dtype=int)
    )
    parent_mira_truth = parent_truth.iloc[parent_positions].copy()
    parent_mira_truth["row_index"] = np.arange(len(parent_mira_truth))
    parent_mira_truth["object_id"] = parent_mira_truth["row_index"]
    parent_mira_truth.to_parquet(
        output / "parent_prior_mira_truth.parquet", index=False
    )
    selected_specs, parent_specs = [], []
    for index, (label, pools) in enumerate(prior_pools.items()):
        selected_path = output / f"prior_{label}_selected_mira.parquet"
        parent_path = output / f"prior_{label}_parent_mira.parquet"
        _prior_mira_frame(
            pools["selected"],
            selected_mira_truth,
            names,
            512,
            manifest["seed"] + 100 + index,
        ).to_parquet(selected_path, index=False)
        _prior_mira_frame(
            pools["parent"],
            parent_mira_truth,
            names,
            512,
            manifest["seed"] + 200 + index,
        ).to_parquet(parent_path, index=False)
        selected_specs.append((f"prior_{label}_selected", selected_path))
        parent_specs.append((f"prior_{label}_parent", parent_path))
    evaluate_feniks_mira(
        truth_path=output / "selected_prior_mira_truth.parquet",
        posterior_specs=selected_specs,
        out_dir=output / "selected_prior_mira",
        samples_per_object=512,
        seed=manifest["seed"] + 300,
        num_regions=100,
        num_bootstrap=1000,
    )
    evaluate_feniks_mira(
        truth_path=output / "parent_prior_mira_truth.parquet",
        posterior_specs=parent_specs,
        out_dir=output / "parent_prior_mira",
        samples_per_object=512,
        seed=manifest["seed"] + 400,
        num_regions=100,
        num_bootstrap=1000,
    )
    prior_mira = []
    for population in ("selected", "parent"):
        frame = pd.read_csv(output / f"{population}_prior_mira/mira_scores.csv")
        frame.insert(0, "population", population)
        prior_mira.append(frame)
    pd.concat(prior_mira, ignore_index=True).to_csv(
        output / "prior_mira_scores.csv", index=False
    )
    write(
        output / "PRIOR_MIRA_CONTRACT.json",
        {
            "selected": "calibration of beta(theta) p_eta(theta) / alpha against selected-catalog truth",
            "parent": "calibration of p_eta(theta) against parent C0 truth",
            "not_posterior_mira": True,
            "truth_used_for_training_or_selection": False,
        },
    )

    selected_series, parent_series = {}, {}
    representative_rows = _representative_rows(inference_truth)
    individual_banks: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    support_rows = []
    population_rows = []
    joint_rows = []
    for index, variant in enumerate(manifest["variants"]):
        label = variant["name"]
        frame = pd.concat(
            [
                pd.read_parquet(path)
                for path in sorted((root / "arms" / label / "bank_0").glob("*.parquet"))
            ],
            ignore_index=True,
        )
        selected_weight, parent_weight, diagnostics = population_weights(frame)
        values = frame[names].to_numpy(np.float64)
        selected_draw = _weighted_draw(
            values, selected_weight, 8192, manifest["seed"] + 500 + index
        )
        parent_draw = _weighted_draw(
            values, parent_weight, 8192, manifest["seed"] + 600 + index
        )
        selected_series[label] = selected_draw
        parent_series[label] = parent_draw
        individual_banks[label] = {}
        frame_rows = frame.row_index.to_numpy()
        for row_index in representative_rows:
            mask = frame_rows == row_index
            local_weight = selected_weight[mask]
            if not mask.any() or not np.isfinite(local_weight).all():
                raise ValueError(f"missing individual bank: {label}/{row_index}")
            local_weight = local_weight / local_weight.sum()
            individual_banks[label][row_index] = (
                frame.loc[mask, names].to_numpy(np.float64),
                local_weight,
            )
        support_rows.append({"variant": label, **diagnostics})
        for population, inferred, truth in (
            (
                "selected_posterior_aggregate",
                selected_draw,
                selected_truth[names].to_numpy(),
            ),
            (
                "inverse_beta_parent_aggregate",
                parent_draw,
                parent_truth[names].to_numpy(),
            ),
        ):
            for parameter_index, parameter in enumerate(names):
                population_rows.append(
                    {
                        "source": label,
                        "population": population,
                        "parameter": parameter,
                        **distribution_comparison(
                            inferred[:, parameter_index], truth[:, parameter_index]
                        ),
                    }
                )
        joint_rows.append(
            {
                **_joint_metric_row(
                    f"{label}_selected_5d",
                    selected_draw[:, :5],
                    selected_truth[physical].to_numpy(),
                ),
                "dimensions": "physical_5d",
            }
        )
        del frame, values, selected_weight, parent_weight
        joint_rows.append(
            {
                **_joint_metric_row(
                    f"{label}_parent_5d",
                    parent_draw[:, :5],
                    parent_truth[physical].to_numpy(),
                ),
                "dimensions": "physical_5d",
            }
        )
    for label, pools in prior_pools.items():
        for population, inferred, truth in (
            ("parent_prior", pools["parent"], parent_truth[names].to_numpy()),
            ("selected_prior", pools["selected"], selected_truth[names].to_numpy()),
        ):
            for parameter_index, parameter in enumerate(names):
                population_rows.append(
                    {
                        "source": f"prior_{label}",
                        "population": population,
                        "parameter": parameter,
                        **distribution_comparison(
                            inferred[:, parameter_index], truth[:, parameter_index]
                        ),
                    }
                )
    pd.DataFrame(support_rows).to_csv(
        output / "population_weight_support.csv", index=False
    )
    pd.DataFrame(population_rows).to_csv(
        output / "population_distribution_metrics.csv", index=False
    )
    pd.DataFrame(joint_rows).to_csv(
        output / "joint_5d_distribution_metrics.csv", index=False
    )

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    _corner(
        plots / "selected_posterior_population_corner_5d.png",
        {key: value[:, :5] for key, value in selected_series.items()},
        selected_truth[physical].to_numpy(),
        physical,
    )
    _corner(
        plots / "inverse_beta_parent_population_corner_5d.png",
        {key: value[:, :5] for key, value in parent_series.items()},
        parent_truth[physical].to_numpy(),
        physical,
    )
    _corner(
        plots / "parent_prior_corner_5d.png",
        {f"prior_{key}": value["parent"][:, :5] for key, value in prior_pools.items()},
        parent_truth[physical].to_numpy(),
        physical,
    )
    _corner(
        plots / "selected_prior_corner_5d.png",
        {
            f"prior_{key}": value["selected"][:, :5]
            for key, value in prior_pools.items()
        },
        selected_truth[physical].to_numpy(),
        physical,
    )
    _marginals(
        plots / "selected_posterior_population_marginals_15d.png",
        selected_series,
        selected_truth[names].to_numpy(),
        names,
    )
    _marginals(
        plots / "inverse_beta_parent_population_marginals_15d.png",
        parent_series,
        parent_truth[names].to_numpy(),
        names,
    )
    _marginals(
        plots / "parent_prior_marginals_15d.png",
        {f"prior_{key}": value["parent"] for key, value in prior_pools.items()},
        parent_truth[names].to_numpy(),
        names,
    )
    _marginals(
        plots / "selected_prior_marginals_15d.png",
        {f"prior_{key}": value["selected"] for key, value in prior_pools.items()},
        selected_truth[names].to_numpy(),
        names,
    )

    individual_dir = plots / "individual"
    individual_dir.mkdir(exist_ok=True)
    individual_records = []
    for order, row_index in enumerate(representative_rows, start=1):
        truth_row = inference_truth.loc[inference_truth.row_index.eq(row_index)].iloc[0]
        object_series = {}
        for offset, variant in enumerate(manifest["variants"]):
            label = variant["name"]
            values, local_weight = individual_banks[label][row_index]
            object_series[label] = _weighted_draw(
                values,
                local_weight,
                2048,
                manifest["seed"] + 1000 + order * 20 + offset,
            )
        stem = f"{order:02d}_row_{row_index}"
        core_path = _corner(
            individual_dir / f"{stem}_posterior_corner_5d.png",
            {key: value[:, :5] for key, value in object_series.items()},
            truth_row[physical].to_numpy(np.float64),
            physical,
        )
        full_path = _marginals(
            individual_dir / f"{stem}_posterior_marginals_15d.png",
            object_series,
            truth_row[names].to_numpy(np.float64),
            names,
        )
        individual_records.append(
            {
                "order": order,
                "row_index": row_index,
                "corner_5d": str(core_path.resolve()),
                "marginals_15d": str(full_path.resolve()),
            }
        )
    write(
        output / "INDIVIDUAL_PLOTS.json",
        {
            "status": "COMPLETE",
            "draws": "ordinary-IS joint posterior resamples",
            "truth_overlay": True,
            "plots": individual_records,
        },
    )
    report_artifacts = {
        str(path.relative_to(root)): sha(path)
        for path in output.rglob("*")
        if path.is_file() and path.name != "FINAL.json"
    }
    write(
        output / "FINAL.json",
        {
            "status": "OVERNIGHT_REPORT_COMPLETE",
            "manifest_sha256": sha(root / "MANIFEST.json"),
            "variants": [item["name"] for item in manifest["variants"]],
            "posterior_mira": True,
            "population_prior_mira": True,
            "individual_truth_corners": len(individual_records),
            "population_parent_and_selected_closure": True,
            "point_estimates_used": False,
            "truth_used_for_training_or_inference": False,
            "inverse_beta_caveat": "interpret only with population_weight_support.csv",
            "scientific_promotion": False,
            "artifacts": report_artifacts,
        },
    )
    print(f"Overnight report complete: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare-training",
            "preflight",
            "train",
            "prepare-inference",
            "infer",
            "report",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prior-followup", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--parent-catalog", type=Path)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=8.0)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.mode == "prepare-training":
        if args.source is None or args.prior_followup is None:
            parser.error("prepare-training requires --source and --prior-followup")
        prepare_training(
            args.source.resolve(), args.prior_followup.resolve(), args.root
        )
    elif args.mode in {"preflight", "train"}:
        from scripts.feniks_avi_experiments import run

        run(args)
    elif args.mode == "prepare-inference":
        if args.training is None or args.source is None or args.prior_followup is None:
            parser.error(
                "prepare-inference requires --training, --source and --prior-followup"
            )
        prepare_inference(
            args.training.resolve(),
            args.source.resolve(),
            args.prior_followup.resolve(),
            args.root,
            parent_catalog=args.parent_catalog,
            particles=args.particles,
        )
    elif args.mode == "infer":
        infer(args.root, args.task)
    else:
        report(args.root)


if __name__ == "__main__":
    main()
