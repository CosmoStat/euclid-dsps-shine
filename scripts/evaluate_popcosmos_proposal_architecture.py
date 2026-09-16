#!/usr/bin/env python3
"""Evaluate one self-supervised posterior architecture on replicated SMC banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from diagnose_popcosmos_proposal_expressivity import (
    _importance_frame,
    _load_banks,
    _load_batch,
    _support_summary,
    _target_evaluator,
)

from euclid_dsps.amortized.config import require_equinox
from euclid_dsps.amortized.posterior import posterior_log_prob, sample_posterior
from euclid_dsps.amortized.proposal_architecture import (
    BandTokenContextEncoder,
    ContextualFlowProposal,
    ResidualContextEncoder,
    contextual_log_prob,
    fit_contextual_proposal,
    make_band_token_observations,
    make_direct_observations,
    sample_contextual_proposal,
)
from euclid_dsps.amortized.proposal_expressivity import (
    count_parameters,
    joint_distribution_metrics,
)
from euclid_dsps.amortized.train import (
    _latent_spec_for_amortized_config,
    load_checkpoint,
)
from euclid_dsps.config import load_config

eqx = require_equinox()


PHASE0_CANDIDATES = (
    "current_compressed",
    "oracle_kde",
    "free_context_rqspline",
    "direct_context_realnvp",
)
PHASE1_CANDIDATES = (
    "current_compressed",
    "direct_context_realnvp",
    "direct_context_rqspline_medium",
    "direct_context_rqspline_large",
    "band_token_rqspline",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--bank-a", type=Path, required=True)
    parser.add_argument("--bank-b", type=Path, required=True)
    parser.add_argument("--phase", choices=("phase0", "phase1"), required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--object-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--proposal-samples", type=int, default=2048)
    parser.add_argument("--decoder-sample-chunk-size", type=int, default=1)
    parser.add_argument("--geometry-draws", type=int, default=256)
    parser.add_argument("--geometry-projections", type=int, default=64)
    parser.add_argument("--kde-centers", type=int, default=128)
    parser.add_argument("--kde-bandwidth", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=260820)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed = PHASE0_CANDIDATES if args.phase == "phase0" else PHASE1_CANDIDATES
    if args.candidate not in allowed:
        raise ValueError(f"candidate {args.candidate!r} is not valid for {args.phase}")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("proposal architecture experiment requires a JAX GPU")
    for path in (
        args.config,
        args.dataset,
        args.checkpoint,
        args.feature_stats,
        args.panel,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for root in (args.bank_a, args.bank_b):
        if not (root / "DONE").is_file():
            raise FileNotFoundError(f"incomplete SMC bank: {root}")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    (args.out / "checkpoints").mkdir()

    config = load_config(args.config)
    config["catalog_path"] = str(args.dataset)
    latent_spec = _latent_spec_for_amortized_config(config)
    model = load_checkpoint(args.checkpoint, config)
    panel = pd.read_parquet(args.panel).sort_values("row_index").reset_index(drop=True)
    required = {"row_index", "expressivity_split", "panel_role"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel missing columns: {missing}")
    rows_a, particles_a, weights_a = _load_banks([args.bank_a], latent_spec.names)
    rows_b, particles_b, weights_b = _load_banks([args.bank_b], latent_spec.names)
    if not np.array_equal(rows_a, rows_b):
        raise ValueError("replicated SMC banks use different object cohorts")
    if not np.array_equal(rows_a, panel["row_index"].to_numpy(dtype=np.int64)):
        panel = panel.set_index("row_index").loc[rows_a].reset_index()
    batch = _load_batch(config, args.feature_stats, row_indices=rows_a)
    train_indices = np.flatnonzero(panel["expressivity_split"].to_numpy() == "train")
    validation_indices = np.flatnonzero(
        panel["expressivity_split"].to_numpy() == "validation"
    )
    print(
        "[proposal-architecture] "
        f"phase={args.phase} candidate={args.candidate} seed={args.seed} "
        f"objects={len(rows_a)} train={len(train_indices)} "
        f"validation={len(validation_indices)} bank_particles={len(particles_a)}",
        flush=True,
    )

    fit_summary = None
    history = None
    checkpoint_path = None
    if args.candidate == "current_compressed":
        sample = sample_posterior(
            model,
            jax.random.PRNGKey(args.seed + 1),
            batch.features,
            args.proposal_samples,
        )
        proposal_x = np.asarray(jax.device_get(sample.x))
        proposal_logq = np.asarray(jax.device_get(sample.logq))
        target_logq = _evaluate_logq_in_chunks(
            lambda value: posterior_log_prob(model, batch.features, value), particles_b
        )
        parameter_count = count_parameters(model.encoder)
    elif args.candidate == "oracle_kde":
        centers, scale = _build_kde(
            particles_a,
            weights_a,
            n_centers=args.kde_centers,
            bandwidth=args.kde_bandwidth,
            seed=args.seed,
        )
        proposal_x, proposal_logq = _sample_kde(
            centers,
            scale,
            n_samples=args.proposal_samples,
            seed=args.seed + 2,
        )
        target_logq = _kde_log_prob(particles_b, centers, scale)
        parameter_count = int(np.prod(centers.shape) + np.prod(scale.shape))
        fit_summary = {
            "method": "weighted SMC-A particle KDE with fixed diagonal bandwidth",
            "n_centers": int(len(centers)),
            "bandwidth": float(args.kde_bandwidth),
            "validation_bank": "independent SMC-B replicate",
        }
    else:
        proposal, observations, fit_indices = _make_candidate(
            args.candidate,
            seed=args.seed,
            features=batch.features,
            mask=batch.mask,
            latent_dim=len(latent_spec.names),
            train_indices=train_indices,
            validation_indices=validation_indices,
        )
        fit = fit_contextual_proposal(
            proposal,
            observations=observations,
            train_particles=particles_a,
            train_weights=weights_a,
            validation_particles=particles_b,
            validation_weights=weights_b,
            train_indices=fit_indices,
            validation_indices=validation_indices,
            epochs=args.epochs,
            object_batch_size=args.object_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            progress_label=args.candidate,
        )
        proposal = fit.proposal
        sample_x, sample_logq = sample_contextual_proposal(
            proposal,
            jax.random.PRNGKey(args.seed + 3),
            observations,
            args.proposal_samples,
        )
        proposal_x = np.asarray(jax.device_get(sample_x))
        proposal_logq = np.asarray(jax.device_get(sample_logq))
        target_logq = _evaluate_logq_in_chunks(
            lambda value: contextual_log_prob(proposal, observations, value),
            particles_b,
        )
        checkpoint_path = args.out / "checkpoints/proposal.eqx"
        eqx.tree_serialise_leaves(checkpoint_path, proposal)
        history = pd.DataFrame(fit.history)
        history.to_csv(args.out / "fit_history.csv", index=False)
        fit_summary = {
            "initial_train_weighted_nll": fit.initial_train_nll,
            "initial_validation_weighted_nll": fit.initial_validation_nll,
            "selected_train_weighted_nll": fit.best_train_nll,
            "best_validation_weighted_nll": fit.best_validation_nll,
            "best_epoch": fit.best_epoch,
            "fit_object_contract": (
                "all panel objects receive free contexts"
                if args.candidate == "free_context_rqspline"
                else "train split only; validation objects unseen during fitting"
            ),
        }
        parameter_count = count_parameters(proposal)

    print("[proposal-architecture] evaluating exact target", flush=True)
    target_evaluator = _target_evaluator(config, model, latent_spec)
    logprior, loglike = target_evaluator(
        proposal_x,
        batch,
        sample_chunk_size=args.decoder_sample_chunk_size,
        progress_label=args.candidate,
    )
    importance = _importance_frame(
        rows_a,
        logprior=logprior,
        loglike=loglike,
        logq=proposal_logq,
    )
    weighted_nll = -np.sum(weights_b * target_logq, axis=0)
    geometry_rows = []
    for object_index, row_index in enumerate(rows_a):
        geometry_rows.append(
            {
                "row_index": int(row_index),
                **joint_distribution_metrics(
                    particles_b[:, object_index],
                    weights_b[:, object_index],
                    proposal_x[:, object_index],
                    seed=args.seed + 1000 + object_index,
                    max_draws=args.geometry_draws,
                    n_projections=args.geometry_projections,
                ),
            }
        )
    objects = importance.merge(pd.DataFrame(geometry_rows), on="row_index")
    objects["weighted_smc_b_nll"] = weighted_nll
    objects["candidate"] = args.candidate
    objects["phase"] = args.phase
    objects["seed"] = int(args.seed)
    objects = objects.merge(
        panel[["row_index", "panel_role", "expressivity_split"]],
        on="row_index",
        validate="one_to_one",
    )
    objects.to_parquet(args.out / "objectwise_diagnostics.parquet", index=False)
    objects.to_csv(args.out / "objectwise_diagnostics.csv", index=False)
    support = _support_summary(objects, min_ess=0.05, max_bad_k=0.2)
    validation = objects.loc[objects["expressivity_split"] == "validation"]
    geometry = {
        key: float(validation[key].median())
        for key in (
            "sliced_wasserstein",
            "energy_distance",
            "nearest_cover_ratio",
            "covariance_relative_frobenius",
        )
    }
    summary = {
        "status": "complete",
        "phase": args.phase,
        "candidate": args.candidate,
        "seed": int(args.seed),
        "self_supervision_contract": (
            "no catalog truth; fit uses normalized weighted SMC-A joint particles, "
            "selection uses independent SMC-B particles and observed photometry"
        ),
        "frozen_contract": (
            "population prior, floor-0.05 Student-t likelihood, feature stats, "
            "object panel, SMC banks and ordinary-IS thresholds are fixed"
        ),
        "n_objects": int(len(rows_a)),
        "n_train": int(len(train_indices)),
        "n_validation": int(len(validation_indices)),
        "parameter_count": int(parameter_count),
        "fit": fit_summary,
        "ordinary_is": support,
        "validation_geometry_medians": geometry,
        "validation_weighted_smc_b_nll": float(validation["weighted_smc_b_nll"].mean()),
        "inputs": {
            "config": _receipt(args.config),
            "dataset": _receipt(args.dataset),
            "checkpoint": _receipt(args.checkpoint),
            "feature_stats": _receipt(args.feature_stats),
            "panel": _receipt(args.panel),
            "bank_a": str(args.bank_a),
            "bank_b": str(args.bank_b),
        },
        "artifacts": {
            "objectwise_diagnostics": _receipt(
                args.out / "objectwise_diagnostics.parquet"
            ),
            "checkpoint": _receipt(checkpoint_path) if checkpoint_path else None,
        },
    }
    (args.out / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out / "DONE").touch()
    validation_support = support["validation"]
    print(
        "[proposal-architecture] "
        f"complete candidate={args.candidate} "
        f"ESS={validation_support['median_raw_ess_fraction']:.6f} "
        f"bad_k={validation_support['fraction_pareto_k_gt_0p7']:.6f} "
        f"support={validation_support['support_status']} -> {args.out}",
        flush=True,
    )


def _make_candidate(
    name,
    *,
    seed,
    features,
    mask,
    latent_dim,
    train_indices,
    validation_indices,
):
    key_context, key_flow = jax.random.split(jax.random.PRNGKey(int(seed)))
    n_objects = len(features)
    if name == "free_context_rqspline":
        observations = jnp.eye(n_objects, dtype=jnp.float32)
        context_dim, context_width, context_depth = 128, 256, 2
        context = ResidualContextEncoder(
            key_context,
            input_dim=n_objects,
            context_dim=context_dim,
            hidden_size=context_width,
            depth=context_depth,
        )
        family, n_layers, hidden_size = "rq_spline", 8, 256
        fit_indices = np.arange(n_objects, dtype=np.int64)
    elif name == "band_token_rqspline":
        observations = make_band_token_observations(features, mask)
        context_dim = 128
        context = BandTokenContextEncoder(
            key_context,
            n_bands=mask.shape[-1],
            token_dim=128,
            context_dim=context_dim,
        )
        family, n_layers, hidden_size = "rq_spline", 8, 256
        fit_indices = train_indices
    else:
        observations = make_direct_observations(features, mask)
        if name == "direct_context_realnvp":
            context_dim, context_width, context_depth = 128, 128, 3
            family, n_layers, hidden_size = "realnvp", 4, 128
        elif name == "direct_context_rqspline_medium":
            context_dim, context_width, context_depth = 128, 256, 3
            family, n_layers, hidden_size = "rq_spline", 8, 256
        elif name == "direct_context_rqspline_large":
            context_dim, context_width, context_depth = 256, 256, 4
            family, n_layers, hidden_size = "rq_spline", 12, 256
        else:
            raise ValueError(f"unknown contextual candidate: {name}")
        context = ResidualContextEncoder(
            key_context,
            input_dim=observations.shape[-1],
            context_dim=context_dim,
            hidden_size=context_width,
            depth=context_depth,
        )
        fit_indices = train_indices
    proposal = ContextualFlowProposal(
        key_flow,
        context_encoder=context,
        context_dim=context_dim,
        latent_dim=latent_dim,
        family=family,
        n_layers=n_layers,
        hidden_size=hidden_size,
        n_bins=8,
        init_scale=0.0,
    )
    if not len(validation_indices):
        raise ValueError("validation split is empty")
    return proposal, observations, np.asarray(fit_indices, dtype=np.int64)


def _evaluate_logq_in_chunks(function, particles, chunk_size=128):
    values = []
    for start in range(0, len(particles), int(chunk_size)):
        values.append(
            np.asarray(
                jax.device_get(
                    function(jnp.asarray(particles[start : start + chunk_size]))
                )
            )
        )
    return np.concatenate(values)


def _build_kde(particles, weights, *, n_centers, bandwidth, seed):
    if not 0.0 < float(bandwidth):
        raise ValueError("KDE bandwidth must be positive")
    particles = np.asarray(particles, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float64)
    n_centers = min(int(n_centers), len(particles))
    rng = np.random.default_rng(int(seed))
    centers = []
    for object_index in range(particles.shape[1]):
        probability = weights[:, object_index] / weights[:, object_index].sum()
        indices = rng.choice(len(particles), n_centers, replace=True, p=probability)
        centers.append(particles[indices, object_index])
    centers = np.stack(centers, axis=1)
    mean = np.sum(weights[..., None] * particles, axis=0)
    variance = np.sum(weights[..., None] * (particles - mean) ** 2, axis=0)
    scale = np.maximum(float(bandwidth) * np.sqrt(variance), 0.03).astype(np.float32)
    return centers, scale


def _sample_kde(centers, scale, *, n_samples, seed):
    key_index, key_noise = jax.random.split(jax.random.PRNGKey(int(seed)))
    centers_jax = jnp.asarray(centers)
    scale_jax = jnp.asarray(scale)
    n_objects = centers.shape[1]
    index = jax.random.randint(
        key_index,
        (int(n_samples), n_objects),
        minval=0,
        maxval=len(centers),
    )
    object_index = jnp.arange(n_objects)[None, :]
    selected = centers_jax[index, object_index]
    values = selected + scale_jax * jax.random.normal(key_noise, selected.shape)
    return np.asarray(values), _kde_log_prob(np.asarray(values), centers, scale)


def _kde_log_prob(values, centers, scale, chunk_size=64):
    values = jnp.asarray(values, dtype=jnp.float32)
    centers = jnp.asarray(centers, dtype=jnp.float32)
    scale = jnp.asarray(scale, dtype=jnp.float32)
    constant = jnp.log(jnp.asarray(2.0 * math.pi, dtype=values.dtype))

    @jax.jit
    def evaluate(chunk):
        standardized = (chunk[:, None, ...] - centers[None, ...]) / scale
        component = -0.5 * jnp.sum(
            standardized**2 + 2.0 * jnp.log(scale) + constant, axis=-1
        )
        return jax.scipy.special.logsumexp(component, axis=1) - jnp.log(
            jnp.asarray(len(centers), dtype=component.dtype)
        )

    result = []
    for start in range(0, len(values), int(chunk_size)):
        result.append(
            np.asarray(jax.device_get(evaluate(values[start : start + chunk_size])))
        )
    return np.concatenate(result)


def _receipt(path):
    if path is None:
        return None
    path = Path(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    main()
