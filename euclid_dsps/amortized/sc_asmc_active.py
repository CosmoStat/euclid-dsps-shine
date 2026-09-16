"""One bounded active bootstrap for a failed SC-ASMC-EM budget preflight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.io import ensure_dir, write_json

from .adaptive_smc_training import make_pmap_e_step, snapshot_model
from .data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from .hierarchical_e_step import (
    _replicate_model,
    _shard_object_batch,
    _slice_smc_result,
    _unshard_smc_result,
)
from .posterior_bank import (
    C0_SCOPE_STATEMENT,
    POSTERIOR_METHOD_CODES,
    PosteriorBankShard,
    merge_posterior_bank_shards,
    read_posterior_bank_shard,
    sha256_file,
    validate_posterior_bank_shard_provenance,
    write_posterior_bank_shard,
)
from .sc_asmc_config import (
    sc_asmc_em_hierarchy,
    sc_asmc_em_schedule,
    validate_sc_asmc_em_config,
)
from .sc_asmc_distill import distill_q_from_full_bank
from .sc_asmc_em import HierarchyDispatch, select_active_bootstrap_rows
from .sc_asmc_estep import posterior_bank_provenance
from .sc_asmc_training import (
    RuntimeBundle,
    load_sc_model,
    validate_component_checkpoint,
)
from .train import _loss_batch


def run_bounded_active_bootstrap(
    config: dict[str, Any],
    runtime: RuntimeBundle,
    run_manifest: dict[str, Any],
    *,
    failed_preflight_bank_root: str | Path,
    q_checkpoint: str | Path,
    q_ema_checkpoint: str | Path,
    prior_checkpoint: str | Path,
    out_dir: str | Path,
    seed: int,
    resume: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build an extended-SMC teacher bank and distill q exactly once."""
    validate_sc_asmc_em_config(config)
    output = ensure_dir(out_dir)
    receipt_path = output / "active_bootstrap_receipt.json"
    checkpoint_hashes = {
        "input_q_checkpoint_sha256": sha256_file(q_checkpoint),
        "input_q_ema_checkpoint_sha256": sha256_file(q_ema_checkpoint),
        "input_prior_checkpoint_sha256": sha256_file(prior_checkpoint),
    }
    validate_component_checkpoint(
        q_checkpoint, checkpoint_hashes["input_q_checkpoint_sha256"], runtime
    )
    validate_component_checkpoint(
        q_ema_checkpoint,
        checkpoint_hashes["input_q_ema_checkpoint_sha256"],
        runtime,
    )
    validate_component_checkpoint(
        prior_checkpoint,
        checkpoint_hashes["input_prior_checkpoint_sha256"],
        runtime,
    )
    preflight_paths = sorted(
        (Path(failed_preflight_bank_root) / "shards").glob("shard_*")
    )
    if not preflight_paths:
        raise ValueError("active bootstrap requires the failed preflight bank")
    expected_provenance = posterior_bank_provenance(
        config,
        runtime,
        run_manifest,
        q_checkpoint=q_checkpoint,
        q_ema_checkpoint=q_ema_checkpoint,
        prior_checkpoint=prior_checkpoint,
    )
    for path in preflight_paths:
        validate_posterior_bank_shard_provenance(path, expected_provenance)
    shards = [read_posterior_bank_shard(path) for path in preflight_paths]
    rows = np.concatenate([shard.row_index for shard in shards])
    method = np.concatenate([shard.method for shard in shards])
    dispatch = HierarchyDispatch(
        method=method,
        resolved=np.concatenate([shard.resolved for shard in shards]),
        primary_attempted=method != POSTERIOR_METHOD_CODES["IS"],
        fallback_attempted=np.isin(
            method,
            [
                POSTERIOR_METHOD_CODES["fallback SMC"],
                POSTERIOR_METHOD_CODES["extended SMC"],
                POSTERIOR_METHOD_CODES["unresolved"],
            ],
        ),
        extended_attempted=np.isin(
            method,
            [
                POSTERIOR_METHOD_CODES["extended SMC"],
                POSTERIOR_METHOD_CODES["unresolved"],
            ],
        ),
    )
    counts = np.concatenate([shard.particle_count for shard in shards])
    hard_rows = select_active_bootstrap_rows(
        rows,
        dispatch,
        ess_fraction=np.concatenate([shard.ess for shard in shards]) / counts,
        max_weight=np.concatenate([shard.max_weight for shard in shards]),
        stage_count=np.concatenate([shard.stage_count for shard in shards]),
        count=int(sc_asmc_em_schedule(config).active_bootstrap_count),
    )
    input_contract = {
        **checkpoint_hashes,
        "input_preflight_rows_sha256": _array_sha256(rows),
        "input_hard_rows_sha256": _array_sha256(hard_rows),
        "input_seed": int(seed),
    }
    if receipt_path.is_file() and resume:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        _validate_active_bootstrap_resume_inputs(payload, input_contract)
        validate_component_checkpoint(
            payload["q_ema_checkpoint"], payload["q_ema_sha256"], runtime
        )
        validate_component_checkpoint(
            payload["q_raw_checkpoint"], payload["q_raw_sha256"], runtime
        )
        return payload
    teacher_root = output / "extended_teacher_bank"
    teacher_paths = _run_extended_teacher(
        config,
        runtime,
        run_manifest,
        rows=hard_rows,
        bank_root=teacher_root,
        q_checkpoint=q_checkpoint,
        q_ema_checkpoint=q_ema_checkpoint,
        prior_checkpoint=prior_checkpoint,
        seed=int(seed),
        resume=resume,
        verbose=verbose,
    )
    teacher_manifest = merge_posterior_bank_shards(
        teacher_root,
        [str(path) for path in teacher_paths],
        expected_row_indices=hard_rows,
    )
    heldout_count = max(16, int(round(0.10 * len(hard_rows))))
    rng = np.random.default_rng(int(seed) + 77)
    heldout = np.sort(rng.choice(hard_rows, size=heldout_count, replace=False))
    distillation = distill_q_from_full_bank(
        config,
        runtime,
        input_bank_manifest=teacher_root / "posterior_bank_manifest.json",
        heldout_rows=heldout,
        q_checkpoint=q_ema_checkpoint,
        prior_checkpoint=prior_checkpoint,
        out_dir=output / "q_distillation",
        iteration=0,
        seed=int(seed) + 1,
        epochs_override=1,
        batch_size_override=64,
        resume=resume,
        verbose=verbose,
    )
    payload = {
        "status": "PASS",
        "phase": "bounded_active_bootstrap",
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "truth_used": False,
        "attempts_allowed": 1,
        "attempts_used": 1,
        **input_contract,
        "hard_objects": int(len(hard_rows)),
        "hard_rows": hard_rows.tolist(),
        "teacher": "extended SMC K128 max_stages=48",
        "teacher_bank_manifest": str(
            (teacher_root / "posterior_bank_manifest.json").resolve()
        ),
        "teacher_bank_objects": int(teacher_manifest["object_count"]),
        "distillation_ratio": "3 teacher updates : 1 Gaussian sleep update",
        "q_ema_checkpoint": distillation["q_ema_checkpoint"],
        "q_ema_sha256": distillation["q_ema_sha256"],
        "q_raw_checkpoint": distillation["q_raw_checkpoint"],
        "q_raw_sha256": distillation["q_raw_sha256"],
        "next_action": "rerun the identical 512-object preflight exactly once",
    }
    write_json(receipt_path, payload)
    return payload


def _array_sha256(values: np.ndarray) -> str:
    import hashlib

    contiguous = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _validate_active_bootstrap_resume_inputs(
    payload: dict[str, Any],
    input_contract: dict[str, Any],
) -> None:
    if any(payload.get(name) != value for name, value in input_contract.items()):
        raise ValueError("active-bootstrap resume inputs changed")


def _run_extended_teacher(
    config: dict[str, Any],
    runtime: RuntimeBundle,
    run_manifest: dict[str, Any],
    *,
    rows: np.ndarray,
    bank_root: Path,
    q_checkpoint: str | Path,
    q_ema_checkpoint: str | Path,
    prior_checkpoint: str | Path,
    seed: int,
    resume: bool,
    verbose: bool,
) -> list[Path]:
    hierarchy = sc_asmc_em_hierarchy(config)
    model = load_sc_model(
        config,
        runtime,
        q_checkpoint=q_ema_checkpoint,
        prior_checkpoint=prior_checkpoint,
    )
    frozen = snapshot_model(model)
    devices = tuple(jax.local_devices())
    fixed_batch = 64
    if fixed_batch % len(devices):
        raise ValueError("active teacher batch must be divisible by local devices")
    step = make_pmap_e_step(
        latent_spec=runtime.jit_latent_spec,
        context=runtime.context,
        model_args=runtime.model_args,
        parameter_names=runtime.parameter_names,
        likelihood_config=runtime.likelihood_config,
        calibration_config=runtime.calibration_config,
        smc_config=hierarchy.extended,
        proposal_config=hierarchy.proposal,
    )
    replicated = _replicate_model(frozen, devices)
    arrays = load_photometry_arrays_from_config(
        runtime.config,
        batch_size=10_000,
        row_indices=np.asarray(rows, dtype=np.int64),
    )
    if arrays.truth:
        raise RuntimeError("active bootstrap loaded truth")
    provenance = posterior_bank_provenance(
        config,
        runtime,
        run_manifest,
        q_checkpoint=q_checkpoint,
        q_ema_checkpoint=q_ema_checkpoint,
        prior_checkpoint=prior_checkpoint,
    )
    paths = []
    for batch_index, photometry in enumerate(
        iter_photometry_batches_from_arrays(
            arrays,
            batch_size=fixed_batch,
            feature_stats=runtime.feature_stats,
            truth_names=None,
        )
    ):
        count = int(photometry.features.shape[0])
        loss_batch = _loss_batch(photometry)
        if count < fixed_batch:
            indices = np.arange(fixed_batch, dtype=np.int64) % count
            take_indices = jnp.asarray(indices)
            loss_batch = jax.tree_util.tree_map(
                lambda value, idx=take_indices: jnp.take(value, idx, axis=0),
                loss_batch,
            )
        sharded = _shard_object_batch(loss_batch, len(devices))
        result = _slice_smc_result(
            _unshard_smc_result(
                step(
                    replicated,
                    sharded,
                    jax.random.split(
                        jax.random.fold_in(jax.random.PRNGKey(seed), batch_index),
                        len(devices),
                    ),
                )
            ),
            count,
        )
        shard = _extended_result_to_bank_shard(
            result,
            model=frozen,
            row_index=np.asarray(photometry.row_index, dtype=np.int64),
            object_id=np.asarray(photometry.object_id).astype(str),
            features=np.asarray(jax.device_get(photometry.features)),
            extended_config=hierarchy.extended,
        )
        write_posterior_bank_shard(
            bank_root,
            batch_index,
            shard,
            provenance,
            resume=resume,
        )
        path = bank_root / "shards" / f"shard_{batch_index:05d}"
        paths.append(path)
        if verbose:
            print(
                "[sc-asmc][active] "
                f"teacher_batch={batch_index} rows={count} "
                f"resolved={np.mean(shard.resolved):.3f}",
                flush=True,
            )
    return paths


def _extended_result_to_bank_shard(
    result: Any,
    *,
    model: Any,
    row_index: np.ndarray,
    object_id: np.ndarray,
    features: np.ndarray,
    extended_config: Any,
) -> PosteriorBankShard:
    particles = np.asarray(jax.device_get(result.final_particles)).transpose(1, 0, 2)
    weights = np.asarray(jax.device_get(result.final_normalized_weights)).T
    source_logprior = np.asarray(
        jax.device_get(model.prior.log_prob(result.final_particles))
    ).T
    hard = np.asarray(jax.device_get(result.hard_object_flag), dtype=bool)
    evaluations = int(extended_config.n_particles) * (
        1
        + np.asarray(jax.device_get(result.number_of_resamples), dtype=np.int64)
        * int(extended_config.steps_after_resample)
        + (np.asarray(jax.device_get(result.beta_final)) >= 1.0 - 1.0e-6).astype(
            np.int64
        )
        * int(extended_config.final_steps_at_beta1)
    )
    method = np.where(
        hard,
        POSTERIOR_METHOD_CODES["unresolved"],
        POSTERIOR_METHOD_CODES["extended SMC"],
    ).astype(np.int8)
    shard = PosteriorBankShard(
        row_index=np.asarray(row_index, dtype=np.int64),
        object_id=np.asarray(object_id, dtype=str),
        method=method,
        particles=particles,
        normalized_weights=weights,
        source_logprior=source_logprior,
        particle_count=np.full(len(row_index), 128, dtype=np.int16),
        ess=np.asarray(jax.device_get(result.final_ess)),
        max_weight=np.asarray(jax.device_get(result.final_max_weight)),
        beta_final=np.asarray(jax.device_get(result.beta_final)),
        logz=np.asarray(jax.device_get(result.logZ_estimate)),
        stage_count=np.asarray(jax.device_get(result.number_of_stages)),
        acceptance=np.asarray(jax.device_get(result.mutation_acceptance)),
        ancestor_ess=np.asarray(jax.device_get(result.ancestor_ess)),
        unique_ancestor_fraction=np.asarray(
            jax.device_get(result.unique_ancestor_fraction)
        ),
        movement_squared=np.asarray(jax.device_get(result.median_epsilon_squared_jump)),
        moved_particle_fraction=np.asarray(
            jax.device_get(result.moved_particle_fraction)
        ),
        dsps_evaluations=evaluations,
        resolved=~hard,
        features=np.asarray(features, dtype=np.float32),
    )
    shard.validate()
    return shard
