"""Post-freeze NUTS cohort and truth-closure utilities for SC-ASMC-EM."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.config import load_config

from .data import load_photometry_arrays_from_config
from .latent import x_to_theta
from .mira import FENIKS_SPLINE15D_PARAMETERS, evaluate_feniks_mira
from .posterior import sample_posterior
from .posterior_bank import (
    C0_SCOPE_STATEMENT,
    POSTERIOR_METHOD_NAMES,
    TARGET_POPULATION_CONTRACT,
    iter_posterior_bank_shards,
    sha256_file,
)
from .sc_asmc_closure_analysis import write_closure_analysis
from .sc_asmc_training import load_sc_model, prepare_sc_runtime
from .tarp import evaluate_feniks_tarp
from .train import _latent_spec_for_amortized_config


def validate_postfreeze_gate(run_root: str | Path) -> dict[str, Any]:
    """Require the hash-bound no-truth final receipt before truth or NUTS."""
    root = Path(run_root)
    receipt_path = root / "FINAL_RECEIPT.json"
    marker_path = root / "FINAL_PASS"
    if not receipt_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError(
            "post-freeze work requires FINAL_RECEIPT.json and FINAL_PASS"
        )
    receipt_hash = sha256_file(receipt_path)
    if marker_path.read_text(encoding="utf-8").strip() != receipt_hash:
        raise ValueError("FINAL_PASS does not bind the current final receipt")
    receipt = _read_json(receipt_path)
    if receipt.get("status") != "PASS":
        raise ValueError("post-freeze work requires a PASS final receipt")
    if receipt.get("c0_scope_statement") != C0_SCOPE_STATEMENT:
        raise ValueError("final receipt lacks the canonical C0 scope")
    if receipt.get("target_population") != TARGET_POPULATION_CONTRACT:
        raise ValueError("final receipt target is not p_eta(theta | C0)")
    no_truth = receipt.get("no_truth_training") or {}
    if no_truth.get("truth_used") is not False:
        raise ValueError("final receipt does not prove no-truth training")
    frozen = receipt.get("frozen_model") or {}
    if frozen.get("model_components") != "q1_ema + p2":
        raise ValueError("final receipt does not identify the q1 EMA + p2 model")
    for name in ("checkpoint", "sidecar"):
        record = frozen.get(name) or {}
        path = Path(record.get("path", ""))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"frozen final model {name} hash mismatch")
    return receipt


def choose_postfreeze_nuts_records(
    records: Sequence[dict[str, Any]],
    *,
    count: int,
    row_indices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Choose a deterministic method- and difficulty-spanning validation cohort."""
    if not 4 <= int(count) <= 8:
        raise ValueError("post-freeze NUTS cohort size must be between 4 and 8")
    resolved = [record for record in records if bool(record["resolved"])]
    by_row = {int(record["row_index"]): record for record in resolved}
    if row_indices is not None:
        requested = [int(value) for value in row_indices]
        if len(requested) != int(count) or len(set(requested)) != len(requested):
            raise ValueError("explicit NUTS rows must be unique and match --count")
        missing = sorted(set(requested) - set(by_row))
        if missing:
            raise ValueError(f"explicit NUTS rows are absent or unresolved: {missing}")
        return [by_row[value] for value in requested]
    if len(resolved) < int(count):
        raise ValueError("final bank has too few resolved objects for NUTS validation")

    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    for method in range(4):
        candidates = [record for record in resolved if int(record["method"]) == method]
        if not candidates:
            continue
        candidates.sort(
            key=lambda value: (float(value["difficulty"]), int(value["row_index"]))
        )
        candidate = candidates[len(candidates) // 2]
        chosen.append(candidate)
        used.add(int(candidate["row_index"]))
        if len(chosen) == int(count):
            return chosen

    remaining = [
        record
        for record in sorted(
            resolved,
            key=lambda value: (float(value["difficulty"]), int(value["row_index"])),
        )
        if int(record["row_index"]) not in used
    ]
    needed = int(count) - len(chosen)
    positions = np.linspace(0, len(remaining) - 1, needed, dtype=int)
    chosen.extend(remaining[int(index)] for index in positions)
    return chosen


def prepare_postfreeze_nuts_cohort(
    run_root: str | Path,
    out_dir: str | Path,
    *,
    count: int = 8,
    row_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Write the exact-benchmark cohort only after the training freeze gate."""
    root = Path(run_root)
    receipt = validate_postfreeze_gate(root)
    bank_path = _final_bank_path(receipt)
    records = _bank_object_records(bank_path)
    chosen = choose_postfreeze_nuts_records(
        records,
        count=int(count),
        row_indices=row_indices,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cohort_path = out / "cohort.parquet"
    contract_path = out / "postfreeze_nuts_cohort_receipt.json"
    if cohort_path.is_file() and contract_path.is_file():
        existing = _read_json(contract_path)
        if existing.get("final_receipt_sha256") != sha256_file(
            root / "FINAL_RECEIPT.json"
        ):
            raise ValueError("existing NUTS cohort belongs to another frozen run")
        if int(existing.get("objects", -1)) != int(count):
            raise ValueError("existing NUTS cohort size differs from --count")
        if row_indices is not None and existing.get("rows") != [
            int(value) for value in row_indices
        ]:
            raise ValueError("existing NUTS cohort rows differ from --rows")
        if sha256_file(cohort_path) != existing.get("cohort", {}).get("sha256"):
            raise ValueError("existing NUTS cohort hash mismatch")
        return existing
    if cohort_path.exists() or contract_path.exists():
        raise FileExistsError("post-freeze NUTS cohort is only partially written")
    rows = []
    for order, record in enumerate(chosen):
        method_name = POSTERIOR_METHOD_NAMES[int(record["method"])]
        rows.append(
            {
                "order": order,
                "example_key": f"{method_name.lower().replace(' ', '_')}_{order:02d}",
                "row_index": int(record["row_index"]),
                "object_id": str(record["object_id"]),
                "posterior_method": method_name,
                "bank_ess_fraction": float(record["ess_fraction"]),
                "bank_max_weight": float(record["max_weight"]),
            }
        )
    frame = pd.DataFrame(rows)
    _atomic_parquet(cohort_path, frame)
    _atomic_csv(out / "cohort.csv", rows)
    payload = {
        "status": "PASS",
        "phase": "postfreeze_nuts_cohort",
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "target_population": TARGET_POPULATION_CONTRACT,
        "training_frozen_before_nuts": True,
        "truth_used": False,
        "nuts_in_training": False,
        "objects": len(rows),
        "rows": [int(row["row_index"]) for row in rows],
        "final_receipt_sha256": sha256_file(root / "FINAL_RECEIPT.json"),
        "final_bank_manifest": str(bank_path.resolve()),
        "final_bank_manifest_sha256": sha256_file(bank_path),
        "cohort": _file_record(cohort_path),
    }
    _atomic_json(contract_path, payload)
    return payload


def run_sc_asmc_truth_closure(
    *,
    training_config_path: str | Path,
    truth_config_path: str | Path,
    run_root: str | Path,
    out_dir: str | Path,
    samples_per_object: int = 128,
    num_mira_regions: int = 100,
    num_bootstrap: int = 1000,
    evaluation_limit: int | None = None,
    seed: int = 260824,
) -> dict[str, Any]:
    """Run truth-aware dense-draw closure after all checkpoints are frozen."""
    if int(samples_per_object) < 2:
        raise ValueError("closure requires at least two dense draws per object")
    root = Path(run_root)
    final = validate_postfreeze_gate(root)
    training_path = Path(training_config_path).resolve()
    truth_path = Path(truth_config_path).resolve()
    if training_path == truth_path:
        raise ValueError("truth closure requires a config separate from training")
    training_config = load_config(training_path)
    truth_config = load_config(truth_path)
    if (training_config.get("truth", {}) or {}).get("parameter_columns"):
        raise ValueError(
            "the training config passed to closure contains truth mappings"
        )
    parameters = tuple(FENIKS_SPLINE15D_PARAMETERS)
    mappings = (truth_config.get("truth", {}) or {}).get("parameter_columns") or {}
    if set(mappings) != set(parameters):
        raise ValueError(
            "truth closure config must map all and only spline15d parameters"
        )
    manifest = _read_json(root / "manifest" / "run_manifest.json")
    dataset = Path(manifest["dataset"]["path"])
    configured_dataset = str(truth_config["catalog_path"])
    if sha256_file(dataset) != manifest["dataset"]["sha256"]:
        raise ValueError("truth closure dataset differs from the frozen run manifest")
    # The immutable run manifest, not a possibly relative YAML default, owns the
    # data identity. Only the separate truth-column mapping comes from closure config.
    truth_config["catalog_path"] = str(dataset.resolve())

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty closure output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    arrays = load_photometry_arrays_from_config(
        truth_config,
        batch_size=int(
            (truth_config.get("amortized", {}) or {})
            .get("data", {})
            .get("catalog_batch_size", 10_000)
        ),
    )
    if arrays.truth is None or set(arrays.truth) != set(parameters):
        raise ValueError("truth-aware loader did not return the exact closure contract")
    if arrays.row_index is None:
        raise ValueError("truth closure requires stable parquet row indices")
    catalogue_rows = np.asarray(arrays.row_index, dtype=np.int64)
    if len(np.unique(catalogue_rows)) != len(catalogue_rows):
        raise ValueError("truth closure catalogue row indices are not unique")
    row_lookup = {int(row): index for index, row in enumerate(catalogue_rows)}
    truth_matrix = np.column_stack(
        [np.asarray(arrays.truth[name]) for name in parameters]
    )
    if not np.all(np.isfinite(truth_matrix)):
        raise ValueError("truth closure contains non-finite physical parameters")
    latent_spec = _latent_spec_for_amortized_config(training_config)
    selected_catalogue_rows = np.load(
        manifest["artifacts"]["selected_rows"]["path"], allow_pickle=False
    )
    selected_catalogue_indices = np.asarray(
        [row_lookup[int(row)] for row in selected_catalogue_rows], dtype=np.int64
    )
    truth_selected_catalog = truth_matrix[selected_catalogue_indices]

    bank_path = _final_bank_path(final)
    em1_bank_path = Path(final["posterior_banks"]["em1"]["path"])
    runtime = prepare_sc_runtime(
        training_config,
        root / ".runtime_cache" / "truth_closure",
        feature_train_rows=manifest["artifacts"]["feature_train_rows"]["path"],
        heldout_rows=manifest["artifacts"]["heldout_rows"]["path"],
    )
    q0_checkpoint = _q0_checkpoint(root)
    q1_checkpoint = final["checkpoints"]["q1_ema"]["path"]
    p0_checkpoint = final["checkpoints"]["p0"]["path"]
    p2_checkpoint = final["checkpoints"]["p2"]["path"]
    q0_model = load_sc_model(
        training_config,
        runtime,
        q_checkpoint=q0_checkpoint,
        prior_checkpoint=p0_checkpoint,
    )
    q1_model = load_sc_model(
        training_config,
        runtime,
        q_checkpoint=q1_checkpoint,
        prior_checkpoint=p2_checkpoint,
    )
    posterior_dir = out / "posterior_samples"
    posterior_dir.mkdir(parents=True, exist_ok=True)
    method_dirs = {
        name: out / "posterior_samples_all_methods" / name
        for name in ("q0", "smc_em1", "q1", "smc_em2")
    }
    for directory in method_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    truth_frames: list[pd.DataFrame] = []
    bias_chunks: list[np.ndarray] = []
    coverage_chunks: dict[float, list[np.ndarray]] = {
        level: [] for level in (0.50, 0.68, 0.90, 0.95)
    }
    posterior_records = []
    all_method_records: dict[str, list[dict[str, Any]]] = {
        name: [] for name in method_dirs
    }
    draw_chunks: dict[str, list[np.ndarray]] = {name: [] for name in method_dirs}
    selected_truth_chunks: list[np.ndarray] = []
    selected_row_chunks: list[np.ndarray] = []
    selected_id_chunks: list[np.ndarray] = []
    selected_catalogue_index_chunks: list[np.ndarray] = []
    excluded_unresolved = 0
    final_shards = iter_posterior_bank_shards(bank_path)
    em1_shards = iter_posterior_bank_shards(em1_bank_path)
    for shard_index, (shard, em1_shard) in enumerate(
        zip(final_shards, em1_shards, strict=True)
    ):
        if not np.array_equal(shard.row_index, em1_shard.row_index):
            raise ValueError("EM1 and final bank shard rows differ")
        selected = np.flatnonzero(np.asarray(shard.resolved, dtype=bool))
        excluded_unresolved += int(shard.object_count - len(selected))
        if not len(selected):
            continue
        rows = np.asarray(shard.row_index[selected], dtype=np.int64)
        try:
            catalogue_index = np.asarray([row_lookup[int(row)] for row in rows])
        except KeyError as error:
            raise ValueError(
                f"posterior row is absent from truth catalogue: {error}"
            ) from error
        object_id = np.asarray(shard.object_id[selected]).astype(str)
        truth_object_id = np.asarray(arrays.object_id[catalogue_index]).astype(str)
        if not np.array_equal(object_id, truth_object_id):
            raise ValueError(
                "posterior-bank object IDs do not match closure truth rows"
            )
        smc2_x = dense_weighted_particle_draws(
            np.asarray(shard.particles[selected]),
            np.asarray(shard.normalized_weights[selected]),
            np.asarray(shard.particle_count[selected]),
            samples=int(samples_per_object),
            seed=int(seed),
            row_indices=rows,
        )
        smc2_theta = np.asarray(
            jax.device_get(x_to_theta(jnp.asarray(smc2_x), latent_spec)),
            dtype=np.float32,
        )
        smc1_x = dense_weighted_particle_draws(
            np.asarray(em1_shard.particles[selected]),
            np.asarray(em1_shard.normalized_weights[selected]),
            np.asarray(em1_shard.particle_count[selected]),
            samples=int(samples_per_object),
            seed=int(seed) + 10_000,
            row_indices=rows,
        )
        smc1_theta = np.asarray(
            jax.device_get(x_to_theta(jnp.asarray(smc1_x), latent_spec)),
            dtype=np.float32,
        )
        features = jnp.asarray(np.asarray(shard.features)[selected])
        q0_x = np.asarray(
            jax.device_get(
                sample_posterior(
                    q0_model,
                    jax.random.fold_in(
                        jax.random.PRNGKey(int(seed) + 20_000), shard_index
                    ),
                    features,
                    int(samples_per_object),
                ).x
            )
        ).transpose(1, 0, 2)
        q1_x = np.asarray(
            jax.device_get(
                sample_posterior(
                    q1_model,
                    jax.random.fold_in(
                        jax.random.PRNGKey(int(seed) + 30_000), shard_index
                    ),
                    features,
                    int(samples_per_object),
                ).x
            )
        ).transpose(1, 0, 2)
        q0_theta = np.asarray(
            jax.device_get(x_to_theta(jnp.asarray(q0_x), latent_spec)), dtype=np.float32
        )
        q1_theta = np.asarray(
            jax.device_get(x_to_theta(jnp.asarray(q1_x), latent_spec)), dtype=np.float32
        )
        method_theta = {
            "q0": q0_theta,
            "smc_em1": smc1_theta,
            "q1": q1_theta,
            "smc_em2": smc2_theta,
        }
        truth_values = truth_matrix[catalogue_index].astype(np.float32)
        posterior_frame = {
            "object_id": np.repeat(object_id, int(samples_per_object)),
            "row_index": np.repeat(rows, int(samples_per_object)),
            "sample_id": np.tile(np.arange(int(samples_per_object)), len(rows)),
        }
        flattened = smc2_theta.reshape(-1, len(parameters))
        posterior_frame.update(
            {name: flattened[:, index] for index, name in enumerate(parameters)}
        )
        shard_path = posterior_dir / f"posterior_samples_{shard_index:05d}.parquet"
        _atomic_parquet(shard_path, pd.DataFrame(posterior_frame))
        posterior_records.append(_file_record(shard_path))
        for method, values in method_theta.items():
            method_frame = {
                "object_id": np.repeat(object_id, int(samples_per_object)),
                "row_index": np.repeat(rows, int(samples_per_object)),
                "sample_id": np.tile(np.arange(int(samples_per_object)), len(rows)),
                **{
                    name: values.reshape(-1, len(parameters))[:, index]
                    for index, name in enumerate(parameters)
                },
            }
            method_path = (
                method_dirs[method] / f"posterior_samples_{shard_index:05d}.parquet"
            )
            _atomic_parquet(method_path, pd.DataFrame(method_frame))
            all_method_records[method].append(_file_record(method_path))
            draw_chunks[method].append(values)
        selected_truth_chunks.append(truth_values)
        selected_row_chunks.append(rows)
        selected_id_chunks.append(object_id)
        selected_catalogue_index_chunks.append(catalogue_index)
        print(
            "[sc-asmc][closure] "
            f"shard={shard_index} resolved_objects={len(rows)} "
            "methods=q0,smc_em1,q1,smc_em2",
            flush=True,
        )
        truth_frames.append(
            pd.DataFrame(
                {
                    "object_id": object_id,
                    "row_index": rows,
                    **{
                        name: truth_values[:, index]
                        for index, name in enumerate(parameters)
                    },
                }
            )
        )
        bias_chunks.append(np.mean(smc2_theta, axis=1) - truth_values)
        for level in coverage_chunks:
            tail = (1.0 - level) / 2.0
            lower, upper = np.quantile(smc2_theta, (tail, 1.0 - tail), axis=1)
            coverage_chunks[level].append(
                (truth_values >= lower) & (truth_values <= upper)
            )
    if not truth_frames:
        raise ValueError("final posterior bank has no resolved closure object")

    inference_truth = pd.concat(truth_frames, ignore_index=True)
    if inference_truth["row_index"].duplicated().any():
        raise ValueError("closure posterior bank contains duplicate row indices")
    truth_output = out / "inference_truth.parquet"
    _atomic_parquet(truth_output, inference_truth)
    coverage_path = out / "marginal_coverage.csv"
    _atomic_csv(
        coverage_path,
        _coverage_rows(coverage_chunks, parameters),
    )
    bias_path = out / "posterior_bias.csv"
    _atomic_csv(bias_path, _bias_rows(np.concatenate(bias_chunks), parameters))

    population_truth_path = out / "population_truth_C0.parquet"
    population_truth = pd.DataFrame(
        {
            "object_id": np.asarray(arrays.object_id).astype(str),
            "row_index": catalogue_rows,
            **{name: truth_matrix[:, index] for index, name in enumerate(parameters)},
        }
    )
    _atomic_parquet(population_truth_path, population_truth)
    prior_artifacts = {
        name: Path(final["report"]["artifacts"][f"prior_{name}"]["path"])
        for name in ("p0", "p1", "p2")
    }
    resolved_catalogue_indices = np.concatenate(selected_catalogue_index_chunks)
    r_index = tuple(arrays.band_names).index("lsst_r")
    r_flux = np.asarray(arrays.flux)[resolved_catalogue_indices, r_index]
    r_error = np.asarray(arrays.flux_err)[resolved_catalogue_indices, r_index]
    closure_analysis = write_closure_analysis(
        out / "analysis",
        draws={
            name: np.concatenate(chunks, axis=0) for name, chunks in draw_chunks.items()
        },
        truth_selected=np.concatenate(selected_truth_chunks, axis=0),
        truth_selected_catalog=truth_selected_catalog,
        row_indices=np.concatenate(selected_row_chunks),
        object_ids=np.concatenate(selected_id_chunks),
        truth_c0=truth_matrix,
        prior_artifacts=prior_artifacts,
        parameters=parameters,
        observed_covariates={
            "r_magnitude_observed": -2.5
            * np.log10(np.maximum(r_flux, np.finfo(np.float64).tiny))
            - 48.6,
            "r_snr": r_flux / np.maximum(r_error, np.finfo(np.float64).tiny),
        },
    )
    print(
        "[sc-asmc][closure] dense mixtures, population comparisons, PIT, "
        "photo-z, coverage, and plots complete",
        flush=True,
    )
    p2_record = final["report"]["artifacts"]["prior_p2"]
    p2_path = Path(p2_record["path"])
    if sha256_file(p2_path) != p2_record["sha256"]:
        raise ValueError("frozen p2 population artifact hash mismatch")
    with np.load(p2_path, allow_pickle=False) as prior_arrays:
        prior_theta = np.asarray(prior_arrays["theta"], dtype=np.float64)
    population_path = out / "population_recovery.csv"
    _atomic_csv(
        population_path,
        _population_recovery_rows(truth_matrix, prior_theta, parameters),
    )
    population_geometry = out / "population_geometry.npz"
    _atomic_npz(
        population_geometry,
        {
            "truth_C0_correlation": np.corrcoef(truth_matrix, rowvar=False),
            "p2_parent_correlation": np.corrcoef(prior_theta, rowvar=False),
            "truth_C0_quantiles": np.quantile(
                truth_matrix, np.linspace(0, 1, 1001), axis=0
            ),
            "p2_parent_quantiles": np.quantile(
                prior_theta, np.linspace(0, 1, 1001), axis=0
            ),
        },
    )

    posterior_specs = tuple(
        (name, method_dirs[name]) for name in ("q0", "smc_em1", "q1", "smc_em2")
    )
    print("[sc-asmc][closure] starting MIRA and TARP", flush=True)
    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="closure-calibration"
    ) as pool:
        mira_future = pool.submit(
            evaluate_feniks_mira,
            truth_path=truth_output,
            posterior_specs=posterior_specs,
            out_dir=out / "mira",
            num_regions=int(num_mira_regions),
            num_bootstrap=int(num_bootstrap),
            samples_per_object=int(samples_per_object),
            seed=int(seed) + 1,
            limit=evaluation_limit,
            parameters=parameters,
        )
        tarp_future = pool.submit(
            evaluate_feniks_tarp,
            truth_path=truth_output,
            posterior_specs=posterior_specs,
            out_dir=out / "tarp",
            num_bootstrap=int(num_bootstrap),
            samples_per_object=int(samples_per_object),
            seed=int(seed) + 2,
            limit=evaluation_limit,
            parameters=parameters,
        )
        mira = mira_future.result()
        tarp = tarp_future.result()
    print("[sc-asmc][closure] MIRA and TARP complete", flush=True)
    payload = {
        "status": "PASS",
        "phase": "postfreeze_truth_closure",
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "target_population": TARGET_POPULATION_CONTRACT,
        "training_frozen_before_truth": True,
        "truth_used": True,
        "truth_used_for_training_or_checkpoint_selection": False,
        "final_receipt_sha256": sha256_file(root / "FINAL_RECEIPT.json"),
        "closure_code_commit": _git_commit(),
        "training_config": _file_record(training_path),
        "truth_only_config": _file_record(truth_path),
        "truth_config_catalog_before_manifest_binding": configured_dataset,
        "truth_catalog_bound_from_frozen_manifest": str(dataset.resolve()),
        "dataset": _file_record(dataset),
        "posterior_bank": _file_record(bank_path),
        "resolved_objects": int(len(inference_truth)),
        "excluded_unresolved_objects": int(excluded_unresolved),
        "dense_draws_per_object": int(samples_per_object),
        "distribution_contract": (
            "equal draws per object preserve every q0, SMC EM1, q1, and SMC EM2 "
            "posterior distribution; parent priors and beta-weighted selected priors "
            "remain separate from selected-catalog posterior mixtures"
        ),
        "artifacts": {
            "inference_truth": _file_record(truth_output),
            "posterior_samples": posterior_records,
            "posterior_samples_all_methods": all_method_records,
            "marginal_coverage": _file_record(coverage_path),
            "posterior_bias": _file_record(bias_path),
            "population_truth_C0": _file_record(population_truth_path),
            "population_recovery": _file_record(population_path),
            "population_geometry": _file_record(population_geometry),
            "mira_manifest": _file_record(out / "mira" / "mira_manifest.json"),
            "tarp_manifest": _file_record(out / "tarp" / "tarp_manifest.json"),
            "closure_analysis": {
                name: _file_record(path)
                if path.is_file()
                else {"path": str(path.resolve()), "kind": "directory"}
                for name, path in closure_analysis.items()
            },
        },
        "mira": mira,
        "tarp": tarp,
    }
    _atomic_json(out / "truth_closure_receipt.json", payload)
    return payload


def dense_weighted_particle_draws(
    particles: np.ndarray,
    normalized_weights: np.ndarray,
    particle_count: np.ndarray,
    *,
    samples: int,
    seed: int,
    row_indices: np.ndarray,
) -> np.ndarray:
    """Draw reproducible dense joint samples without point-estimate collapse."""
    values = np.asarray(particles)
    weights = np.asarray(normalized_weights, dtype=np.float64)
    counts = np.asarray(particle_count, dtype=np.int64)
    rows = np.asarray(row_indices, dtype=np.int64)
    if values.ndim != 3 or weights.shape != values.shape[:2]:
        raise ValueError("weighted particles must be [objects, particles, latent]")
    if counts.shape != (values.shape[0],) or rows.shape != counts.shape:
        raise ValueError("particle counts and row indices must be object vectors")
    result = np.empty(
        (values.shape[0], int(samples), values.shape[2]), dtype=values.dtype
    )
    for index, (count, row) in enumerate(zip(counts, rows, strict=True)):
        active = weights[index, : int(count)]
        if not np.isclose(np.sum(active), 1.0, atol=1.0e-8, rtol=1.0e-6):
            raise ValueError("closure weights are not normalized")
        rng = np.random.default_rng(np.random.SeedSequence((int(seed), int(row))))
        choice = rng.choice(int(count), size=int(samples), replace=True, p=active)
        result[index] = values[index, choice]
    return result


def _q0_checkpoint(root: Path) -> str:
    active = root / "preflight" / "active_bootstrap" / "active_bootstrap_receipt.json"
    if active.is_file():
        return str(_read_json(active)["q_ema_checkpoint"])
    sleep = _read_json(root / "sleep" / "sleep_receipt.json")
    return str(sleep["q_ema_checkpoint"])


def _final_bank_path(receipt: dict[str, Any]) -> Path:
    banks = receipt.get("posterior_banks") or {}
    record = banks.get("em2_p2_repaired") or banks.get("em2_p2")
    if not isinstance(record, dict) or not record.get("path"):
        raise ValueError("final receipt does not identify a final posterior bank")
    return Path(record["path"])


def _bank_object_records(bank_path: Path) -> list[dict[str, Any]]:
    records = []
    for shard in iter_posterior_bank_shards(bank_path):
        for index, row in enumerate(shard.row_index):
            count = int(shard.particle_count[index])
            ess_fraction = float(shard.ess[index]) / count
            max_weight = float(shard.max_weight[index])
            records.append(
                {
                    "row_index": int(row),
                    "object_id": str(shard.object_id[index]),
                    "method": int(shard.method[index]),
                    "resolved": bool(shard.resolved[index]),
                    "ess_fraction": ess_fraction,
                    "max_weight": max_weight,
                    "difficulty": max_weight - ess_fraction,
                }
            )
    return records


def _coverage_rows(
    chunks: dict[float, list[np.ndarray]],
    parameters: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for level, values in chunks.items():
        matrix = np.concatenate(values, axis=0)
        for index, name in enumerate(parameters):
            fraction = float(np.mean(matrix[:, index]))
            rows.append(
                {
                    "parameter": name,
                    "nominal_coverage": float(level),
                    "empirical_coverage": fraction,
                    "coverage_error": fraction - float(level),
                    "objects": int(len(matrix)),
                }
            )
    return rows


def _bias_rows(values: np.ndarray, parameters: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "parameter": name,
            "mean_posterior_mean_minus_truth": float(np.mean(values[:, index])),
            "median_posterior_mean_minus_truth": float(np.median(values[:, index])),
            "rmse_posterior_mean": float(np.sqrt(np.mean(values[:, index] ** 2))),
            "objects": int(len(values)),
        }
        for index, name in enumerate(parameters)
    ]


def _population_recovery_rows(
    truth: np.ndarray,
    prior: np.ndarray,
    parameters: Sequence[str],
) -> list[dict[str, Any]]:
    probabilities = np.linspace(0.0, 1.0, 1001)
    truth_quantiles = np.quantile(truth, probabilities, axis=0)
    prior_quantiles = np.quantile(prior, probabilities, axis=0)
    return [
        {
            "parameter": name,
            "truth_C0_objects": int(len(truth)),
            "p2_parent_samples": int(len(prior)),
            "wasserstein_1d": float(
                np.mean(np.abs(truth_quantiles[:, index] - prior_quantiles[:, index]))
            ),
            "mean_difference": float(
                np.mean(prior[:, index]) - np.mean(truth[:, index])
            ),
            "std_ratio": float(np.std(prior[:, index]) / np.std(truth[:, index])),
            "q05_difference": float(
                prior_quantiles[50, index] - truth_quantiles[50, index]
            ),
            "q50_difference": float(
                prior_quantiles[500, index] - truth_quantiles[500, index]
            ),
            "q95_difference": float(
                prior_quantiles[950, index] - truth_quantiles[950, index]
            ),
        }
        for index, name in enumerate(parameters)
    ]


def _file_record(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty post-freeze CSV")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
