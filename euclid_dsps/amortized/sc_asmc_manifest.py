"""Observed-only catalogue manifest for the FENIKS SC-ASMC-EM workflow."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from euclid_dsps.photometry import abmag_to_fnu_cgs

from .posterior_bank import (
    C0_SCOPE_STATEMENT,
    OBSERVED_SELECTION_CONTRACT,
    TARGET_POPULATION_CONTRACT,
    sha256_file,
)
from .sc_asmc_config import sc_asmc_em_config_hash, validate_sc_asmc_em_config
from .sc_asmc_em import stratified_preflight_indices


def prepare_sc_asmc_manifest(
    config: dict[str, Any],
    out_dir: str | Path,
    *,
    catalogue_path: str | Path | None = None,
    n_estep_shards: int = 4,
    seed: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Build immutable row manifests without requesting any truth column."""
    validation = validate_sc_asmc_em_config(config)
    schedule = validation["schedule"]
    config_sha256 = sc_asmc_em_config_hash(config)
    run_seed = int(
        seed
        if seed is not None
        else (config.get("amortized", {}) or {}).get("training", {}).get("seed", 260824)
    )
    path = Path(catalogue_path or config["catalog_path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if int(n_estep_shards) <= 0:
        raise ValueError("n_estep_shards must be positive")
    output = Path(out_dir)
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if resume:
            _validate_existing_manifest(
                existing,
                path,
                int(n_estep_shards),
                config_sha256=config_sha256,
                seed=run_seed,
            )
            return existing
        raise FileExistsError(manifest_path)
    output.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(path)
    schema_names = tuple(parquet.schema_arrow.names)
    band_contract = _observed_band_contract(config, schema_names)
    columns = list(dict.fromkeys(band_contract["columns_read"]))
    table = pq.read_table(path, columns=columns)
    flux = np.column_stack(
        [
            np.asarray(table[name], dtype=np.float64)
            for name in band_contract["flux_columns"]
        ]
    )
    flux_err = np.column_stack(
        [
            np.asarray(table[name], dtype=np.float64)
            for name in band_contract["error_columns"]
        ]
    )
    masks = np.ones(flux.shape, dtype=bool)
    for index, name in enumerate(band_contract["mask_columns"]):
        if name is not None:
            masks[:, index] = np.asarray(table[name], dtype=bool)
    object_ids = (
        np.asarray(table[band_contract["object_id_column"]].to_pylist(), dtype=str)
        if band_contract["object_id_column"] is not None
        else np.asarray([str(value) for value in range(len(flux))], dtype=str)
    )
    finite_observation = np.all(
        (~masks) | (np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0.0)),
        axis=1,
    )
    r_index = tuple(band_contract["band_names"]).index("lsst_r")
    flux_limit = float(np.asarray(abmag_to_fnu_cgs(25.0)))
    selected_mask = finite_observation & masks[:, r_index]
    selected_mask &= flux[:, r_index] > flux_limit
    selected = np.flatnonzero(selected_mask).astype(np.int64)
    if len(selected) < int(schedule.preflight_objects):
        raise ValueError(
            "observed r<25 catalogue has fewer than 512 valid selected objects"
        )

    rng = np.random.default_rng(run_seed)
    shuffled = rng.permutation(selected)
    validation_fraction = float(
        (config.get("amortized", {}) or {})
        .get("data", {})
        .get("validation_fraction", 0.10)
    )
    n_heldout = max(1, int(round(validation_fraction * len(selected))))
    n_heldout = min(n_heldout, len(selected) - 1)
    heldout = np.sort(shuffled[:n_heldout])
    feature_train = np.sort(shuffled[n_heldout:])
    preflight = stratified_preflight_indices(
        np.arange(len(flux), dtype=np.int64)[selected],
        flux[selected],
        flux_err[selected],
        r_band_index=r_index,
        flux_limit=flux_limit,
        count=int(schedule.preflight_objects),
        seed=run_seed + 17,
    )
    shard_rows = tuple(
        np.sort(values.astype(np.int64))
        for values in np.array_split(selected, int(n_estep_shards))
    )
    if any(len(values) == 0 for values in shard_rows):
        raise ValueError("E-step shard count exceeds selected object count")

    artifacts: dict[str, Any] = {}
    for name, values in {
        "selected_rows": selected,
        "feature_train_rows": feature_train,
        "heldout_rows": heldout,
        "preflight_rows": preflight,
    }.items():
        artifacts[name] = _write_npy(output / f"{name}.npy", values)
    for shard_id, values in enumerate(shard_rows):
        artifacts[f"estep_shard_{shard_id:02d}"] = _write_npy(
            output / "estep_shards" / f"shard_{shard_id:02d}.npy",
            values,
        )

    upstream = _audit_upstream_provenance(path)
    payload = {
        "status": "complete",
        "schema_version": 1,
        "workflow": "Selection-Corrected Amortized SMC-EM",
        "acronym": "SC-ASMC-EM",
        "config_sha256": config_sha256,
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "target_population": TARGET_POPULATION_CONTRACT,
        "observed_selection": OBSERVED_SELECTION_CONTRACT,
        "scope_limit": "No inference claim is made outside C0.",
        "truth_used_for_training_or_checkpoint_selection": False,
        "truth_columns_requested": [],
        "observed_columns_read": columns,
        "dataset": {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "parquet_rows": int(parquet.metadata.num_rows),
            "schema_names_sha256": _json_sha256(list(schema_names)),
            "upstream_selection_provenance": upstream,
        },
        "code": _git_provenance(),
        "selection": {
            "event": OBSERVED_SELECTION_CONTRACT,
            "band": "lsst_r",
            "max_mag_ab": 25.0,
            "flux_min_fnu_cgs": flux_limit,
            "comparison": "observed flux_lsst_r > f_nu(25 AB)",
            "enters_object_weights": False,
            "selected_objects": int(len(selected)),
        },
        "features": {
            "statistics_source": "observed feature_train_rows only",
            "heldout_source": "observed heldout_rows only",
            "all_band_masks_true": bool(np.all(masks)),
            "append_mask": bool(
                (config.get("amortized", {}) or {})
                .get("features", {})
                .get("append_mask", False)
            ),
            "input_dim": int(validation["input_dim"]),
        },
        "preflight": {
            "objects": int(len(preflight)),
            "strata": [
                "distance from observed r=25 cut",
                "observed r-band SNR",
                "observed error quantiles",
                "observed colours",
            ],
            "scientific_training_subset": False,
        },
        "e_step_shards": {
            "count": int(n_estep_shards),
            "covers_every_selected_object_exactly_once": True,
            "object_counts": [int(len(values)) for values in shard_rows],
        },
        "objects": {
            "selected": int(len(selected)),
            "feature_train": int(len(feature_train)),
            "heldout": int(len(heldout)),
        },
        "artifacts": artifacts,
        "seed": run_seed,
        "object_id_digest_selected": _json_sha256(object_ids[selected].tolist()),
    }
    _atomic_json(manifest_path, payload)
    return payload


def validate_sc_asmc_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = Path(payload["dataset"]["path"])
    if sha256_file(dataset) != payload["dataset"]["sha256"]:
        raise ValueError("SC-ASMC-EM dataset hash changed after manifest creation")
    arrays = {}
    for name, record in payload["artifacts"].items():
        artifact = Path(record["path"])
        if sha256_file(artifact) != record["sha256"]:
            raise ValueError(f"SC-ASMC-EM manifest artifact hash mismatch: {name}")
        arrays[name] = np.load(artifact, allow_pickle=False)
    selected = arrays["selected_rows"]
    shards = [
        arrays[f"estep_shard_{index:02d}"]
        for index in range(int(payload["e_step_shards"]["count"]))
    ]
    merged = np.concatenate(shards)
    if not np.array_equal(np.sort(merged), np.sort(selected)):
        raise ValueError("E-step shards do not cover the selected catalogue")
    if len(np.unique(merged)) != len(merged):
        raise ValueError("E-step shards contain duplicate selected rows")
    if len(arrays["preflight_rows"]) != 512:
        raise ValueError("integrated cost preflight must contain exactly 512 rows")
    if payload.get("truth_columns_requested") != []:
        raise ValueError("manifest requested truth columns")
    return payload


def _observed_band_contract(
    config: dict[str, Any],
    schema_names: tuple[str, ...],
) -> dict[str, Any]:
    available = set(schema_names)
    bands = tuple(config.get("bands", ()))
    if len(bands) != 18:
        raise ValueError("SC-ASMC-EM requires exactly 18 configured bands")
    flux_columns = tuple(str(band["column"]) for band in bands)
    error_columns = tuple(str(band["error_column"]) for band in bands)
    missing = (set(flux_columns) | set(error_columns)) - available
    if missing:
        raise ValueError(f"catalogue is missing observed photometry: {sorted(missing)}")
    mask_columns = tuple(_mask_column(str(band["name"]), available) for band in bands)
    object_id_column = next(
        (name for name in ("object_id", "galaxy_id", "id") if name in available),
        None,
    )
    columns = [*flux_columns, *error_columns]
    columns.extend(name for name in mask_columns if name is not None)
    if object_id_column is not None:
        columns.append(object_id_column)
    return {
        "band_names": tuple(str(band["name"]) for band in bands),
        "flux_columns": flux_columns,
        "error_columns": error_columns,
        "mask_columns": mask_columns,
        "object_id_column": object_id_column,
        "columns_read": tuple(columns),
    }


def _mask_column(band_name: str, available: set[str]) -> str | None:
    candidates = (f"mask_{band_name}", f"{band_name}_mask")
    return next((value for value in candidates if value in available), None)


def _audit_upstream_provenance(catalogue: Path) -> dict[str, Any]:
    candidates = (
        catalogue.parent / "amortized_catalog_contract.json",
        catalogue.parent.parent / "dataset_contract.json",
        catalogue.parent.parent / "manifest.json",
    )
    records = []
    for candidate in candidates:
        if candidate.is_file():
            records.append(
                {
                    "path": str(candidate.resolve()),
                    "sha256": sha256_file(candidate),
                    "size_bytes": candidate.stat().st_size,
                }
            )
    return {
        "status": "audited_from_existing_artifacts",
        "domain": "predefined FENIKS refinement and catalogue-support domain C0",
        "may_include_upstream_true_space_cuts": True,
        "dataset_regenerated": False,
        "supporting_artifacts": records,
    }


def _git_provenance() -> dict[str, Any]:
    """Record the checkout revision and dirty paths without modifying the tree."""
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("SC-ASMC-EM requires git code provenance") from error
    return {
        "repository_root": str(Path(root).resolve()),
        "commit": commit,
        "working_tree_dirty": bool(status),
        "dirty_entries": status,
    }


def _write_npy(path: Path, values: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values, dtype=np.int64), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path.resolve()),
        "count": int(len(values)),
        "sha256": sha256_file(path),
        "minimum": int(np.min(values)),
        "maximum": int(np.max(values)),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _json_sha256(value: Any) -> str:
    import hashlib

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_existing_manifest(
    payload: dict[str, Any],
    catalogue: Path,
    shard_count: int,
    *,
    config_sha256: str,
    seed: int,
) -> None:
    if Path(payload["dataset"]["path"]) != catalogue:
        raise ValueError("existing SC-ASMC-EM manifest uses another catalogue")
    if int(payload["e_step_shards"]["count"]) != int(shard_count):
        raise ValueError("existing SC-ASMC-EM manifest uses another shard count")
    if payload.get("config_sha256") != config_sha256:
        raise ValueError("existing SC-ASMC-EM manifest uses another configuration")
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError("existing SC-ASMC-EM manifest uses another seed")
    validate_sc_asmc_manifest(
        Path(payload["artifacts"]["selected_rows"]["path"]).parent / "run_manifest.json"
    )
