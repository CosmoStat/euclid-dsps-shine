"""Prior-ratio bank reweighting with selective low-ESS posterior refresh."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.io import ensure_dir, write_json

from .data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from .hierarchical_e_step import (
    build_pmap_hierarchy_kernels,
    hierarchy_result_to_bank_shard,
    run_pmap_model_hierarchical_e_step,
)
from .posterior_bank import (
    C0_SCOPE_STATEMENT,
    PosteriorBankShard,
    is_posterior_bank_shard_complete,
    low_reweight_ess_rows,
    read_posterior_bank_shard,
    replace_posterior_bank_rows,
    reweight_posterior_bank_shard,
    sha256_file,
    validate_posterior_bank_manifest_provenance,
    validate_posterior_bank_shard_provenance,
    write_posterior_bank_shard,
)
from .sc_asmc_config import sc_asmc_em_hierarchy, validate_sc_asmc_em_config
from .sc_asmc_estep import posterior_bank_provenance
from .sc_asmc_training import RuntimeBundle, load_sc_model
from .train import _loss_batch


def reweight_and_refresh_bank_worker(
    config: dict[str, Any],
    runtime: RuntimeBundle,
    run_manifest: dict[str, Any],
    *,
    input_bank_manifest: str | Path,
    output_bank_root: str | Path,
    worker_id: int,
    worker_count: int,
    q_checkpoint: str | Path,
    q_ema_checkpoint: str | Path,
    old_prior_checkpoint: str | Path,
    new_prior_checkpoint: str | Path,
    seed: int,
    resume: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Reweight assigned shards and replace only low-ESS object rows."""
    validate_sc_asmc_em_config(config)
    if not 0 <= int(worker_id) < int(worker_count):
        raise ValueError("bank reweight worker_id must be in [0, worker_count)")
    output = ensure_dir(output_bank_root)
    receipt_path = output / f"worker_{int(worker_id):02d}_receipt.json"
    input_manifest = json.loads(Path(input_bank_manifest).read_text(encoding="utf-8"))
    assigned = [
        record
        for index, record in enumerate(input_manifest["shards"])
        if index % int(worker_count) == int(worker_id)
    ]
    if not assigned:
        raise ValueError("reweight worker was assigned no posterior-bank shard")
    expected_old_prior_hash = sha256_file(old_prior_checkpoint)
    expected_source_provenance = posterior_bank_provenance(
        config,
        runtime,
        run_manifest,
        q_checkpoint=q_checkpoint,
        q_ema_checkpoint=q_ema_checkpoint,
        prior_checkpoint=old_prior_checkpoint,
    )
    validate_posterior_bank_manifest_provenance(
        input_manifest,
        expected_fields=asdict(expected_source_provenance),
    )
    expected_new_prior_hash = sha256_file(new_prior_checkpoint)
    provenance = posterior_bank_provenance(
        config,
        runtime,
        run_manifest,
        q_checkpoint=q_checkpoint,
        q_ema_checkpoint=q_ema_checkpoint,
        prior_checkpoint=new_prior_checkpoint,
    )
    if receipt_path.is_file() and resume:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        _validate_reweight_receipt(
            payload,
            provenance=provenance,
            expected_old_prior_hash=expected_old_prior_hash,
            expected_new_prior_hash=expected_new_prior_hash,
            assigned=assigned,
        )
        return payload
    new_model = load_sc_model(
        config,
        runtime,
        q_checkpoint=q_ema_checkpoint,
        prior_checkpoint=new_prior_checkpoint,
    )
    hierarchy = sc_asmc_em_hierarchy(config)
    raw = dict(
        ((config.get("amortized", {}) or {}).get("sc_asmc_em", {}) or {}).get(
            "bank_reweight", {}
        )
        or {}
    )
    minimum_ess_fraction = float(raw.get("minimum_ess_fraction", 0.30))
    devices = tuple(jax.local_devices())
    primary_batch_size = max(32, 32 * int(np.ceil(len(devices) / 32)))
    primary_batch_size = max(len(devices), primary_batch_size)
    if primary_batch_size % len(devices):
        primary_batch_size += len(devices) - primary_batch_size % len(devices)
    kernels = build_pmap_hierarchy_kernels(
        latent_spec=runtime.jit_latent_spec,
        context=runtime.context,
        model_args=runtime.model_args,
        parameter_names=runtime.parameter_names,
        likelihood_config=runtime.likelihood_config,
        calibration_config=runtime.calibration_config,
        primary_config=hierarchy.primary,
        fallback_config=hierarchy.fallback,
        extended_config=hierarchy.extended,
        proposal_config=hierarchy.proposal,
        minimum_is_ess_fraction=hierarchy.minimum_is_ess_fraction,
        maximum_is_weight=hierarchy.maximum_is_weight,
        primary_batch_size=primary_batch_size,
        fallback_batch_size=max(len(devices), 16),
        extended_batch_size=max(len(devices), 16),
        devices=devices,
    )
    shard_records = []
    low_count = 0
    refresh_count = 0
    futures = []
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="bank-reweight") as pool:
        for local_index, record in enumerate(assigned):
            input_shard = read_posterior_bank_shard(record["path"])
            output_shard_id = int(record["shard_id"])
            output_path = output / "shards" / f"shard_{output_shard_id:05d}"
            if resume and is_posterior_bank_shard_complete(
                output_path, validate_arrays=True
            ):
                validate_posterior_bank_shard_provenance(output_path, provenance)
                completed = read_posterior_bank_shard(output_path)
                shard_records.append(
                    {
                        "path": str(output_path.resolve()),
                        "shard_id": output_shard_id,
                        "rows": completed.object_count,
                        "resumed": True,
                    }
                )
                continue
            new_logprior = _evaluate_prior_on_bank(new_model.prior, input_shard)
            reweighted = reweight_posterior_bank_shard(
                input_shard,
                new_logprior,
            )
            low_rows = low_reweight_ess_rows(
                reweighted.row_index,
                reweighted.ess,
                reweighted.particle_count,
                minimum_ess_fraction=minimum_ess_fraction,
            )
            low_count += int(len(low_rows))
            if len(low_rows):
                replacement = _refresh_rows(
                    runtime=runtime,
                    model=new_model,
                    rows=low_rows,
                    kernels=kernels,
                    hierarchy=hierarchy,
                    seed=int(seed) + local_index,
                )
                reweighted = replace_posterior_bank_rows(reweighted, replacement)
                refresh_count += int(len(low_rows))
            futures.append(
                pool.submit(
                    write_posterior_bank_shard,
                    output,
                    output_shard_id,
                    reweighted,
                    provenance,
                    resume=resume,
                )
            )
            shard_records.append(
                {
                    "path": str(output_path.resolve()),
                    "shard_id": output_shard_id,
                    "rows": reweighted.object_count,
                    "low_reweight_ess_rows": int(len(low_rows)),
                    "resumed": False,
                }
            )
            if verbose:
                print(
                    "[sc-asmc][reweight] "
                    f"worker={worker_id} shard={output_shard_id} "
                    f"rows={reweighted.object_count} refreshed={len(low_rows)}",
                    flush=True,
                )
        for future in futures:
            future.result()
    payload = {
        "status": "complete",
        "phase": "bank_reweight_and_selective_refresh",
        "worker_id": int(worker_id),
        "worker_count": int(worker_count),
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "truth_used": False,
        "q_frozen": True,
        "old_prior_frozen_for_ratio": True,
        "ratio": "w_new proportional to w_old * p_new(x) / p_old(x)",
        "old_prior_checkpoint_hash": expected_old_prior_hash,
        "new_prior_checkpoint_hash": expected_new_prior_hash,
        "minimum_reweight_ess_fraction": minimum_ess_fraction,
        "low_ess_rows": low_count,
        "refreshed_rows": refresh_count,
        "extended_applied_uniformly": False,
        "shards": shard_records,
    }
    write_json(receipt_path, payload)
    return payload


def _validate_reweight_receipt(
    payload: dict[str, Any],
    *,
    provenance: Any,
    expected_old_prior_hash: str,
    expected_new_prior_hash: str,
    assigned: list[dict[str, Any]],
) -> None:
    if payload.get("status") != "complete":
        raise ValueError("reweight receipt is incomplete")
    if payload.get("old_prior_checkpoint_hash") != expected_old_prior_hash:
        raise ValueError("reweight resume uses another source prior")
    if payload.get("new_prior_checkpoint_hash") != expected_new_prior_hash:
        raise ValueError("reweight resume uses another destination prior")
    records = payload.get("shards", [])
    if len(records) != len(assigned):
        raise ValueError("reweight receipt shard assignment changed")
    expected_ids = {int(record["shard_id"]) for record in assigned}
    actual_ids = {int(record["shard_id"]) for record in records}
    if actual_ids != expected_ids:
        raise ValueError("reweight receipt does not cover its assigned shards")
    for record in records:
        if not is_posterior_bank_shard_complete(record["path"], validate_arrays=True):
            raise ValueError("reweight receipt references an incomplete shard")
        validate_posterior_bank_shard_provenance(record["path"], provenance)


def _evaluate_prior_on_bank(prior: Any, shard: PosteriorBankShard) -> np.ndarray:
    values = jnp.asarray(shard.particles)
    result = np.zeros(shard.source_logprior.shape, dtype=np.float64)
    for start in range(0, shard.object_count, 64):
        stop = min(start + 64, shard.object_count)
        # Prior accepts arbitrary leading axes; transpose keeps particle-major
        # ordering consistent with the E-step implementation.
        particles = jnp.asarray(values[start:stop]).transpose(1, 0, 2)
        logprior = prior.log_prob(particles)
        result[start:stop] = np.asarray(jax.device_get(logprior)).T
    return result


def _refresh_rows(
    *,
    runtime: RuntimeBundle,
    model: Any,
    rows: np.ndarray,
    kernels: Any,
    hierarchy: Any,
    seed: int,
) -> PosteriorBankShard:
    arrays = load_photometry_arrays_from_config(
        runtime.config,
        batch_size=10_000,
        row_indices=np.asarray(rows, dtype=np.int64),
    )
    if arrays.truth:
        raise RuntimeError("selective posterior refresh loaded truth")
    replacements = []
    for batch_index, photometry in enumerate(
        iter_photometry_batches_from_arrays(
            arrays,
            batch_size=kernels.primary_batch_size,
            feature_stats=runtime.feature_stats,
            truth_names=None,
        )
    ):
        result = run_pmap_model_hierarchical_e_step(
            model_snapshot=model,
            batch=_loss_batch(photometry),
            key=jax.random.fold_in(jax.random.PRNGKey(int(seed)), batch_index),
            kernels=kernels,
        )
        replacements.append(
            hierarchy_result_to_bank_shard(
                result,
                model_snapshot=model,
                row_index=np.asarray(photometry.row_index, dtype=np.int64),
                object_id=np.asarray(photometry.object_id).astype(str),
                features=np.asarray(jax.device_get(photometry.features)),
                particle_capacity=128,
                primary_config=hierarchy.primary,
                fallback_config=hierarchy.fallback,
                extended_config=hierarchy.extended,
            )
        )
    return _concatenate_bank_rows(replacements)


def _concatenate_bank_rows(shards: list[PosteriorBankShard]) -> PosteriorBankShard:
    if not shards:
        raise ValueError("cannot concatenate an empty posterior refresh")
    values = {}
    for name in PosteriorBankShard.__dataclass_fields__:
        if name == "feature_reference":
            values[name] = shards[0].feature_reference
        elif name == "features":
            values[name] = (
                None
                if shards[0].features is None
                else np.concatenate([shard.features for shard in shards], axis=0)
            )
        else:
            values[name] = np.concatenate(
                [np.asarray(getattr(shard, name)) for shard in shards], axis=0
            )
    result = PosteriorBankShard(**values)
    result.validate()
    return result
