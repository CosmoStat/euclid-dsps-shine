#!/usr/bin/env python3
"""Evaluate beta on a bounded subset of existing joint q-bank draws."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from euclid_dsps.amortized.population_projection import (
    evaluate_log_beta,
    selection_runtime,
)
from euclid_dsps.amortized.population_vem import (
    ArrayShardContract,
    read_array_bank_shard,
    require_git_commit,
    resolve_manifest_config,
    sha256_file,
    write_array_bank_shard,
)
from euclid_dsps.amortized.train import load_checkpoint
from euclid_dsps.config import load_config

SELECTION_EVENT = "A=1[m_r_observed<29.0]"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_shard(
    manifest: dict[str, Any], split: str, shard: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    record = manifest["q_banks"][split]
    bank = _read_json(Path(record["manifest"]))
    source_record = bank["shards"][int(shard)]
    source_path = Path(source_record["path"])
    if sha256_file(source_path / "arrays.npz") != source_record["arrays_sha256"]:
        raise ValueError(f"source q-bank shard SHA256 mismatch: {source_path}")
    return read_array_bank_shard(source_path), bank["contract"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = _read_json(root / "RUN_MANIFEST.json")
    repo = Path(__file__).resolve().parents[1]
    require_git_commit(repo, manifest["code_commit"])
    task = int(args.task)
    if 0 <= task < 16:
        split = "fit"
        bank_name = "beta_fit"
        kind = "q_beta_fit"
        shard = task
        requested_draws = int(manifest["resources"]["fit_draws_for_beta"])
    elif 16 <= task < 20:
        split = "validation"
        bank_name = "beta_validation"
        kind = "q_beta_validation"
        shard = task - 16
        requested_draws = int(manifest["resources"]["validation_draws_for_beta"])
    else:
        raise ValueError("population-projection beta task must lie in [0, 19]")

    arrays, source_contract = _source_shard(manifest, split, shard)
    source_draws = int(source_contract["draws_per_object"])
    if requested_draws > source_draws:
        raise ValueError("requested beta draws exceed the source q-bank draw count")
    draw_index = np.linspace(
        0, source_draws, requested_draws, endpoint=False, dtype=np.int64
    )
    x = np.asarray(arrays["x"][:, draw_index, :], dtype=np.float32)
    log_q = np.asarray(arrays["log_q"][:, draw_index], dtype=np.float32)

    checkpoint = Path(manifest["source"]["checkpoint"])
    if sha256_file(checkpoint) != manifest["source"]["checkpoint_sha256"]:
        raise ValueError("source q checkpoint changed before beta evaluation")
    sidecar = Path(manifest["source"]["checkpoint_sidecar"])
    if sha256_file(sidecar) != manifest["source"]["checkpoint_sidecar_sha256"]:
        raise ValueError("source q checkpoint sidecar changed before beta evaluation")
    feature_stats = Path(manifest["source"]["feature_stats"])
    if sha256_file(feature_stats) != manifest["source"]["feature_stats_sha256"]:
        raise ValueError("source feature statistics changed before beta evaluation")
    config = load_config(resolve_manifest_config(manifest, "config", repo))
    model = load_checkpoint(checkpoint, config)
    runtime = selection_runtime(config, feature_stats)
    flat_log_beta = evaluate_log_beta(
        model,
        x.reshape(-1, x.shape[-1]),
        runtime,
        chunk_size=int(args.chunk_size),
    )
    log_beta = flat_log_beta.reshape(x.shape[:2])
    contract = ArrayShardContract(
        kind=kind,
        dataset_sha256=source_contract["dataset_sha256"],
        checkpoint_sha256=source_contract["checkpoint_sha256"],
        latent_transform_sha256=source_contract["latent_transform_sha256"],
        code_commit=manifest["code_commit"],
        truth_used=False,
        draws_per_object=requested_draws,
        feature_stats_sha256=source_contract["feature_stats_sha256"],
        row_indices_sha256=source_contract["row_indices_sha256"],
        selection_event=SELECTION_EVENT,
    )
    receipt = write_array_bank_shard(
        root / "banks" / bank_name,
        shard,
        {
            "row_index": np.asarray(arrays["row_index"], dtype=np.int64),
            "draw_index": draw_index,
            "x": x,
            "log_q": log_q,
            "log_beta": np.asarray(log_beta, dtype=np.float64),
        },
        contract,
    )
    print(
        f"[population-projection-beta] split={split} shard={shard} "
        f"objects={len(arrays['row_index'])} draws={requested_draws} "
        f"arrays_sha256={receipt['arrays_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
