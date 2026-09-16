#!/usr/bin/env python3
"""Evaluate one zero-initialized warm-start posterior context adapter."""

from __future__ import annotations

import argparse
import json
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
from evaluate_popcosmos_proposal_architecture import (
    _evaluate_logq_in_chunks,
    _receipt,
)

from euclid_dsps.amortized.config import require_equinox
from euclid_dsps.amortized.posterior import posterior_log_prob, sample_posterior
from euclid_dsps.amortized.proposal_architecture import (
    FreeResidualContextAdapter,
    WarmStartResidualProposal,
    contextual_log_prob,
    fit_contextual_proposal,
    make_direct_observations,
    sample_contextual_proposal,
    zero_band_token_adapter,
    zero_residual_mlp_adapter,
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

ADAPTER_CANDIDATES = (
    "current_compressed",
    "free_context_adapter",
    "direct_photometry_adapter",
    "band_token_adapter",
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
    parser.add_argument("--candidate", choices=ADAPTER_CANDIDATES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--object-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--proposal-samples", type=int, default=2048)
    parser.add_argument("--decoder-sample-chunk-size", type=int, default=1)
    parser.add_argument("--geometry-draws", type=int, default=256)
    parser.add_argument("--geometry-projections", type=int, default=64)
    parser.add_argument("--seed", type=int, default=260820)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("proposal adapter experiment requires a JAX GPU")
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
        "[proposal-adapter] "
        f"candidate={args.candidate} seed={args.seed} objects={len(rows_a)} "
        f"train={len(train_indices)} validation={len(validation_indices)} "
        f"bank_particles={len(particles_a)} proposal_samples={args.proposal_samples}",
        flush=True,
    )

    fit_summary = None
    checkpoint_path = None
    clone_error = 0.0
    source_change = 0.0
    adapter_parameter_count = 0
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
    else:
        proposal, observations, fit_indices = _make_adapter(
            args.candidate,
            model.encoder,
            features=batch.features,
            mask=batch.mask,
            seed=args.seed,
            train_indices=train_indices,
        )
        initial_current = _evaluate_logq_in_chunks(
            lambda value: posterior_log_prob(model, batch.features, value),
            particles_b[:32],
            chunk_size=32,
        )
        initial_adapter = _evaluate_logq_in_chunks(
            lambda value: contextual_log_prob(proposal, observations, value),
            particles_b[:32],
            chunk_size=32,
        )
        clone_error = float(np.max(np.abs(initial_current - initial_adapter)))
        if clone_error > 2.0e-5:
            raise RuntimeError(
                f"zero adapter does not reproduce current q: max_abs={clone_error}"
            )
        source_before = _array_leaves(proposal.source_encoder)
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
            freeze_source_encoder=True,
        )
        proposal = fit.proposal
        source_after = _array_leaves(proposal.source_encoder)
        source_change = max(
            float(np.max(np.abs(before - after)))
            for before, after in zip(source_before, source_after, strict=True)
        )
        if source_change != 0.0:
            raise RuntimeError(f"frozen source encoder changed: {source_change}")
        sample_x, sample_logq = sample_contextual_proposal(
            proposal,
            jax.random.PRNGKey(args.seed + 1),
            observations,
            args.proposal_samples,
        )
        proposal_x = np.asarray(jax.device_get(sample_x))
        proposal_logq = np.asarray(jax.device_get(sample_logq))
        target_logq = _evaluate_logq_in_chunks(
            lambda value: contextual_log_prob(proposal, observations, value),
            particles_b,
        )
        checkpoint_path = args.out / "checkpoints/adapter.eqx"
        eqx.tree_serialise_leaves(checkpoint_path, proposal)
        pd.DataFrame(fit.history).to_csv(args.out / "fit_history.csv", index=False)
        fit_summary = {
            "initial_train_weighted_nll": fit.initial_train_nll,
            "initial_validation_weighted_nll": fit.initial_validation_nll,
            "selected_train_weighted_nll": fit.best_train_nll,
            "best_validation_weighted_nll": fit.best_validation_nll,
            "best_epoch": fit.best_epoch,
            "fit_object_contract": (
                "all objects receive per-object corrections fitted on SMC-A"
                if args.candidate == "free_context_adapter"
                else "train objects only; validation objects unseen during fitting"
            ),
        }
        parameter_count = count_parameters(proposal)
        adapter_parameter_count = count_parameters(proposal.adapter)

    target_evaluator = _target_evaluator(config, model, latent_spec)
    logprior, loglike = target_evaluator(
        proposal_x,
        batch,
        sample_chunk_size=args.decoder_sample_chunk_size,
        progress_label=args.candidate,
    )
    importance = _importance_frame(
        rows_a, logprior=logprior, loglike=loglike, logq=proposal_logq
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
        "candidate": args.candidate,
        "seed": int(args.seed),
        "proposal_samples_per_object": int(args.proposal_samples),
        "self_supervision_contract": (
            "no catalog truth; fit uses weighted SMC-A joint particles and "
            "selection uses independent SMC-B particles"
        ),
        "warm_start_contract": {
            "initial_max_abs_logq_error": clone_error,
            "frozen_source_encoder_max_abs_change": source_change,
            "source_encoder_frozen_exactly": source_change == 0.0,
        },
        "n_objects": int(len(rows_a)),
        "n_train": int(len(train_indices)),
        "n_validation": int(len(validation_indices)),
        "parameter_count": int(parameter_count),
        "adapter_parameter_count": int(adapter_parameter_count),
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
    (args.out / "adapter_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out / "DONE").touch()
    validation_support = support["validation"]
    print(
        "[proposal-adapter] "
        f"complete candidate={args.candidate} "
        f"ESS={validation_support['median_raw_ess_fraction']:.6f} "
        f"bad_k={validation_support['fraction_pareto_k_gt_0p7']:.6f} "
        f"support={validation_support['support_status']} -> {args.out}",
        flush=True,
    )


def _make_adapter(name, source_encoder, *, features, mask, seed, train_indices):
    direct = make_direct_observations(features, mask)
    feature_dim = int(features.shape[-1])
    context_dim = 2 * int(source_encoder.latent_dim)
    if name == "free_context_adapter":
        n_objects = len(features)
        observations = jnp.concatenate(
            (direct, jnp.eye(n_objects, dtype=jnp.float32)), axis=-1
        )
        adapter = FreeResidualContextAdapter(
            one_hot_offset=direct.shape[-1],
            n_objects=n_objects,
            context_dim=context_dim,
        )
        fit_indices = np.arange(n_objects, dtype=np.int64)
    elif name == "direct_photometry_adapter":
        observations = direct
        adapter = zero_residual_mlp_adapter(
            jax.random.PRNGKey(int(seed)),
            input_dim=observations.shape[-1],
            context_dim=context_dim,
            hidden_size=128,
            depth=3,
        )
        fit_indices = train_indices
    elif name == "band_token_adapter":
        observations = direct
        adapter = zero_band_token_adapter(
            jax.random.PRNGKey(int(seed)),
            n_bands=mask.shape[-1],
            token_dim=128,
            context_dim=context_dim,
        )
        fit_indices = train_indices
    else:
        raise ValueError(f"unknown adapter candidate: {name}")
    return (
        WarmStartResidualProposal(source_encoder, adapter, feature_dim=feature_dim),
        observations,
        np.asarray(fit_indices, dtype=np.int64),
    )


def _array_leaves(value):
    return [
        np.asarray(leaf).copy()
        for leaf in jax.tree.leaves(eqx.filter(value, eqx.is_inexact_array))
        if leaf is not None
    ]


if __name__ == "__main__":
    main()
