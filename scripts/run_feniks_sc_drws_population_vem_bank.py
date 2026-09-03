#!/usr/bin/env python3
"""Build one restartable GPU shard for the population-VEM workflow."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.amortized.data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from euclid_dsps.amortized.features import read_feature_stats
from euclid_dsps.amortized.latent import theta_to_x
from euclid_dsps.amortized.population_vem import (
    ArrayShardContract,
    require_git_commit,
    resolve_manifest_config,
    sha256_file,
    write_array_bank_shard,
)
from euclid_dsps.amortized.posterior import sample_posterior
from euclid_dsps.amortized.train import (
    JitLatentSpec,
    _latent_spec_for_amortized_config,
    _selection_correction_runtime_config,
    _selection_log_beta_from_prior_samples,
    load_checkpoint,
)
from euclid_dsps.config import load_config
from euclid_dsps.filters import load_filters
from euclid_dsps.model import dynamic_model_args, load_context

eqx, _optax = require_amortized_dependencies()

SELECTION_EVENT = "A=1[m_r_observed<29.0]"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _contract(
    manifest: dict[str, Any],
    *,
    kind: str,
    dataset: str,
    checkpoint_sha256: str,
    draws: int | None = None,
    row_sha256: str | None = None,
    truth_used: bool = False,
) -> ArrayShardContract:
    return ArrayShardContract(
        kind=kind,
        dataset_sha256=manifest["datasets"][dataset]["sha256"],
        checkpoint_sha256=checkpoint_sha256,
        latent_transform_sha256=manifest["frozen_source"]["latent_transform_sha256"],
        code_commit=manifest["code_commit"],
        truth_used=truth_used,
        draws_per_object=draws,
        feature_stats_sha256=(
            manifest["frozen_source"]["feature_stats_sha256"]
            if kind.startswith("q_")
            else None
        ),
        row_indices_sha256=row_sha256,
        selection_event=(
            SELECTION_EVENT
            if kind in {"selection_reference", "selection_audit", "prior_evaluation"}
            else None
        ),
    )


def _load_rows(manifest: dict[str, Any], bank: str, shard: int) -> np.ndarray:
    record = manifest["banks"][bank]["records"][int(shard)]
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"row shard SHA256 mismatch: {path}")
    return np.load(path, allow_pickle=False).astype(np.int64)


def _jit_latent_spec(spec) -> JitLatentSpec:
    return JitLatentSpec(
        names=spec.names,
        lower=spec.lower,
        upper=spec.upper,
        raw_center=spec.raw_center,
        raw_scale=spec.raw_scale,
        normalization=spec.normalization,
        transform_family=spec.transform_family,
        transform_location=spec.transform_location,
        transform_lambda=spec.transform_lambda,
    )


def _selection_runtime(config: dict[str, Any], feature_stats_path: Path):
    no_truth = copy.deepcopy(config)
    no_truth.setdefault("truth", {})["parameter_columns"] = {}
    stats = read_feature_stats(feature_stats_path)
    filters = load_filters(no_truth["bands"])
    context = load_context(
        no_truth["ssp_path"],
        filters,
        n_sfh_bins=int(no_truth["model"].get("n_sfh_bins", 96)),
        cosmos_config=no_truth.get("cosmos_sed"),
        nebular_emission=no_truth.get("nebular_emission", "ssp_flux"),
        model_config=no_truth.get("model"),
    )
    latent_spec = _latent_spec_for_amortized_config(no_truth)
    return (
        latent_spec,
        _jit_latent_spec(latent_spec),
        context,
        dynamic_model_args(context),
        _selection_correction_runtime_config(no_truth, stats),
        {"calibration": no_truth.get("calibration", {}) or {}},
    )


def _evaluate_log_beta(
    model,
    x: np.ndarray,
    runtime,
    *,
    chunk_size: int,
) -> np.ndarray:
    latent_spec, jit_spec, context, model_args, selection, calibration = runtime
    pieces = []
    for start in range(0, len(x), int(chunk_size)):
        stop = min(start + int(chunk_size), len(x))
        values = _selection_log_beta_from_prior_samples(
            model,
            jnp.asarray(x[start:stop]),
            jit_spec,
            context,
            model_args,
            latent_spec.names,
            calibration,
            selection,
        )
        pieces.append(np.asarray(jax.device_get(values), dtype=np.float64))
        print(
            f"[population-vem-bank] beta {stop}/{len(x)}",
            flush=True,
        )
    return np.concatenate(pieces)


def _write_q_bank(
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
    checkpoint: Path,
    checkpoint_sha256: str,
    feature_stats_path: Path,
    bank: str,
    kind: str,
    shard: int,
    draws: int,
    dataset: str,
    out: Path,
    seed: int,
) -> None:
    rows = _load_rows(manifest, bank, shard)
    source_config = copy.deepcopy(config)
    source_config["catalog_path"] = manifest["datasets"][dataset]["path"]
    source_config.setdefault("truth", {})["parameter_columns"] = {}
    feature_stats = read_feature_stats(feature_stats_path)
    arrays = load_photometry_arrays_from_config(
        source_config,
        batch_size=10_000,
        row_indices=rows,
    )
    if arrays.truth:
        raise RuntimeError("truth entered a q-bank task")
    model = load_checkpoint(checkpoint, source_config)

    @eqx.filter_jit
    def draw(model, key, features):
        return sample_posterior(model, key, features, int(draws))

    x_parts = []
    log_q_parts = []
    row_parts = []
    key = jax.random.PRNGKey(int(seed))
    for batch_number, batch in enumerate(
        iter_photometry_batches_from_arrays(
            arrays,
            batch_size=256,
            feature_stats=feature_stats,
        ),
        start=1,
    ):
        key, draw_key = jax.random.split(key)
        sample = draw(model, draw_key, batch.features)
        x_parts.append(
            np.asarray(jax.device_get(sample.x), dtype=np.float32).transpose(1, 0, 2)
        )
        log_q_parts.append(
            np.asarray(jax.device_get(sample.logq), dtype=np.float32).transpose(1, 0)
        )
        row_parts.append(np.asarray(batch.row_index, dtype=np.int64))
        print(
            f"[population-vem-bank] {bank} shard={shard} "
            f"batch={batch_number} objects={sum(map(len, row_parts))}/{len(rows)}",
            flush=True,
        )
    contract = _contract(
        manifest,
        kind=kind,
        dataset=dataset,
        checkpoint_sha256=checkpoint_sha256,
        draws=draws,
        row_sha256=manifest["banks"][bank]["cohort_sha256"],
    )
    write_array_bank_shard(
        out / "banks" / bank,
        shard,
        {
            "row_index": np.concatenate(row_parts),
            "x": np.concatenate(x_parts),
            "log_q": np.concatenate(log_q_parts),
        },
        contract,
    )


def _write_selection_reference(
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
    checkpoint: Path,
    checkpoint_sha256: str,
    feature_stats_path: Path,
    shard: int,
    out: Path,
    seed: int,
    bank: str,
    samples: int,
) -> None:
    model = load_checkpoint(checkpoint, config)
    key = jax.random.PRNGKey(int(seed))
    x = np.asarray(jax.device_get(model.prior.sample(key, int(samples))))
    log_p = np.asarray(jax.device_get(model.prior.log_prob(jnp.asarray(x))))
    runtime = _selection_runtime(config, feature_stats_path)
    log_beta = _evaluate_log_beta(model, x, runtime, chunk_size=128)
    kind = (
        "selection_reference" if bank == "selection_reference" else "prior_evaluation"
    )
    contract = _contract(
        manifest,
        kind=kind,
        dataset="train",
        checkpoint_sha256=checkpoint_sha256,
    )
    arrays = {"x": x.astype(np.float32), "log_beta": log_beta.astype(np.float64)}
    if kind == "selection_reference":
        arrays["log_p_reference"] = log_p.astype(np.float32)
    else:
        arrays["log_p"] = log_p.astype(np.float32)
    write_array_bank_shard(out / "banks" / bank, shard, arrays, contract)


def _write_selection_audit(
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
    truth_config: dict[str, Any],
    checkpoint: Path,
    checkpoint_sha256: str,
    feature_stats_path: Path,
    shard: int,
    out: Path,
) -> None:
    rows = _load_rows(manifest, "selection_audit", shard)
    truth_config = copy.deepcopy(truth_config)
    truth_config["catalog_path"] = manifest["datasets"]["train"]["path"]
    arrays = load_photometry_arrays_from_config(
        truth_config,
        batch_size=10_000,
        row_indices=rows,
    )
    latent_spec = _latent_spec_for_amortized_config(config)
    if not arrays.truth or any(name not in arrays.truth for name in latent_spec.names):
        raise RuntimeError("selection audit is missing frozen C0 truth")
    theta = np.column_stack(
        [np.asarray(arrays.truth[name]) for name in latent_spec.names]
    )
    if not np.all(np.isfinite(theta)):
        raise ValueError("selection audit truth contains non-finite parameters")
    x = np.asarray(jax.device_get(theta_to_x(jnp.asarray(theta), latent_spec)))
    model = load_checkpoint(checkpoint, config)
    runtime = _selection_runtime(config, feature_stats_path)
    log_beta = _evaluate_log_beta(model, x, runtime, chunk_size=128)
    selected_rows = np.concatenate(
        (
            np.load(manifest["banks"]["q_fit"]["cohort_path"], allow_pickle=False),
            np.load(
                manifest["banks"]["q_validation"]["cohort_path"],
                allow_pickle=False,
            ),
        )
    )
    selected = np.isin(np.asarray(arrays.row_index), selected_rows)
    contract = _contract(
        manifest,
        kind="selection_audit",
        dataset="train",
        checkpoint_sha256=checkpoint_sha256,
        row_sha256=manifest["banks"]["selection_audit"]["cohort_sha256"],
        truth_used=True,
    )
    write_array_bank_shard(
        out / "banks" / "selection_audit",
        shard,
        {
            "row_index": np.asarray(arrays.row_index, dtype=np.int64),
            "beta": np.exp(log_beta).astype(np.float64),
            "selected": selected.astype(np.bool_),
            "redshift": theta[:, 0].astype(np.float64),
        },
        contract,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("initial", "final"), required=True)
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = _read_json(root / "RUN_MANIFEST.json")
    repo = Path(__file__).resolve().parents[1]
    require_git_commit(repo, manifest["code_commit"])
    config = load_config(resolve_manifest_config(manifest, "config", repo))
    truth_config = load_config(resolve_manifest_config(manifest, "truth_config", repo))
    feature_stats = Path(manifest["frozen_source"]["feature_stats"])
    if sha256_file(feature_stats) != manifest["frozen_source"]["feature_stats_sha256"]:
        raise ValueError("feature-stat provenance changed")

    if args.stage == "initial":
        checkpoint = Path(manifest["frozen_source"]["checkpoint"])
        checkpoint_sha256 = manifest["frozen_source"]["checkpoint_sha256"]
        if sha256_file(checkpoint) != checkpoint_sha256:
            raise ValueError("source checkpoint provenance changed")
        sidecar = Path(manifest["frozen_source"]["checkpoint_sidecar"])
        if (
            sha256_file(sidecar)
            != manifest["frozen_source"]["checkpoint_sidecar_sha256"]
        ):
            raise ValueError("source checkpoint sidecar provenance changed")
        task = int(args.task)
        if 0 <= task < 16:
            _write_q_bank(
                manifest=manifest,
                config=config,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                feature_stats_path=feature_stats,
                bank="q_fit",
                kind="q_train",
                shard=task,
                draws=32,
                dataset="train",
                out=root,
                seed=261000 + task,
            )
        elif 16 <= task < 20:
            shard = task - 16
            _write_q_bank(
                manifest=manifest,
                config=config,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                feature_stats_path=feature_stats,
                bank="q_validation",
                kind="q_validation",
                shard=shard,
                draws=64,
                dataset="train",
                out=root,
                seed=262000 + shard,
            )
        elif 20 <= task < 28:
            shard = task - 20
            _write_selection_reference(
                manifest=manifest,
                config=config,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                feature_stats_path=feature_stats,
                shard=shard,
                out=root,
                seed=263000 + shard,
                bank="selection_reference",
                samples=2048,
            )
        elif 28 <= task < 36:
            _write_selection_audit(
                manifest=manifest,
                config=config,
                truth_config=truth_config,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                feature_stats_path=feature_stats,
                shard=task - 28,
                out=root,
            )
        else:
            raise ValueError("initial bank task must lie in [0, 35]")
    else:
        refresh = _read_json(root / "q_refresh" / "Q_REFRESH_COMPLETE.json")
        if refresh.get("status") != "COMPLETE":
            raise ValueError("final banks require a complete q refresh")
        checkpoint = Path(refresh["checkpoint"])
        checkpoint_sha256 = refresh["checkpoint_sha256"]
        if sha256_file(checkpoint) != checkpoint_sha256:
            raise ValueError("refreshed checkpoint provenance changed")
        checkpoint_sidecar = checkpoint.with_suffix(".eqx.json")
        if sha256_file(checkpoint_sidecar) != refresh["checkpoint_sidecar_sha256"]:
            raise ValueError("refreshed checkpoint sidecar provenance changed")
        task = int(args.task)
        if 0 <= task < 8:
            _write_q_bank(
                manifest=manifest,
                config=config,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                feature_stats_path=feature_stats,
                bank="q_evaluation",
                kind="q_evaluation",
                shard=task,
                draws=32,
                dataset="test",
                out=root,
                seed=264000 + task,
            )
        elif 8 <= task < 16:
            shard = task - 8
            _write_selection_reference(
                manifest=manifest,
                config=config,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                feature_stats_path=feature_stats,
                shard=shard,
                out=root,
                seed=265000 + shard,
                bank="prior_evaluation",
                samples=2048,
            )
        else:
            raise ValueError("final bank task must lie in [0, 15]")
    print(
        f"[population-vem-bank] complete stage={args.stage} task={args.task}",
        flush=True,
    )


if __name__ == "__main__":
    main()
