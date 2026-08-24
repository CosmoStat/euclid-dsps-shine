"""Persistent sharded posterior banks for selection-corrected SMC-EM."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

C0_SCOPE_STATEMENT = (
    "We infer the parent distribution within the predefined FENIKS refinement "
    "and catalogue-support domain, while explicitly correcting the additional "
    "observed r<25 selection."
)
TARGET_POPULATION_CONTRACT = "p_eta(theta | C0)"
OBSERVED_SELECTION_CONTRACT = "A = 1[m_r_observed < 25]"

POSTERIOR_METHOD_CODES = {
    "IS": 0,
    "primary SMC": 1,
    "fallback SMC": 2,
    "extended SMC": 3,
    "unresolved": 4,
}
POSTERIOR_METHOD_NAMES = {value: key for key, value in POSTERIOR_METHOD_CODES.items()}


@dataclass(frozen=True)
class PosteriorBankProvenance:
    dataset_hash: str
    workflow_config_hash: str
    q_checkpoint_hash: str
    q_ema_hash: str
    prior_checkpoint_hash: str
    latent_transform_hash: str
    feature_stats_hash: str
    likelihood_contract: dict[str, Any]
    selection_contract: dict[str, Any]
    code_commit: str
    upstream_selection_provenance: dict[str, Any]
    c0_scope_statement: str = C0_SCOPE_STATEMENT
    target_population: str = TARGET_POPULATION_CONTRACT

    def validate(self) -> None:
        hash_fields = (
            self.dataset_hash,
            self.workflow_config_hash,
            self.q_checkpoint_hash,
            self.q_ema_hash,
            self.prior_checkpoint_hash,
            self.latent_transform_hash,
            self.feature_stats_hash,
        )
        if any(not _is_sha256(value) for value in hash_fields):
            raise ValueError("posterior-bank provenance requires SHA256 hashes")
        if self.c0_scope_statement != C0_SCOPE_STATEMENT:
            raise ValueError("posterior-bank C0 scope statement is not canonical")
        if self.target_population != TARGET_POPULATION_CONTRACT:
            raise ValueError(
                "posterior-bank target population must be p_eta(theta | C0)"
            )
        if self.selection_contract.get("event") != OBSERVED_SELECTION_CONTRACT:
            raise ValueError("posterior-bank selection contract must be observed r<25")
        if self.selection_contract.get("enters_object_weights") is not False:
            raise ValueError("selection must not enter object posterior weights")
        if self.likelihood_contract.get("family") not in {"gaussian", "normal"}:
            raise ValueError("main posterior bank requires the Gaussian likelihood")
        if not self.code_commit:
            raise ValueError("posterior-bank provenance requires a code commit")


@dataclass(frozen=True)
class PosteriorBankShard:
    """One object-major bank shard with fixed padded particle capacity."""

    row_index: np.ndarray
    object_id: np.ndarray
    method: np.ndarray
    particles: np.ndarray
    normalized_weights: np.ndarray
    source_logprior: np.ndarray
    particle_count: np.ndarray
    ess: np.ndarray
    max_weight: np.ndarray
    beta_final: np.ndarray
    logz: np.ndarray
    stage_count: np.ndarray
    acceptance: np.ndarray
    ancestor_ess: np.ndarray
    unique_ancestor_fraction: np.ndarray
    movement_squared: np.ndarray
    moved_particle_fraction: np.ndarray
    dsps_evaluations: np.ndarray
    resolved: np.ndarray
    features: np.ndarray | None = None
    feature_reference: str | None = None

    @property
    def object_count(self) -> int:
        return int(np.asarray(self.row_index).shape[0])

    @property
    def particle_capacity(self) -> int:
        return int(np.asarray(self.particles).shape[1])

    @property
    def latent_dim(self) -> int:
        return int(np.asarray(self.particles).shape[2])

    def validate(self) -> None:
        n_objects = self.object_count
        particles = np.asarray(self.particles)
        if particles.ndim != 3 or particles.shape[0] != n_objects:
            raise ValueError(
                "bank particles must have shape [objects, particles, latent]"
            )
        capacity = int(particles.shape[1])
        matrix_fields = {
            "normalized_weights": self.normalized_weights,
            "source_logprior": self.source_logprior,
        }
        for name, value in matrix_fields.items():
            if np.asarray(value).shape != (n_objects, capacity):
                raise ValueError(f"bank {name} must have shape {(n_objects, capacity)}")
        object_fields = (
            "object_id",
            "method",
            "particle_count",
            "ess",
            "max_weight",
            "beta_final",
            "logz",
            "stage_count",
            "acceptance",
            "ancestor_ess",
            "unique_ancestor_fraction",
            "movement_squared",
            "moved_particle_fraction",
            "dsps_evaluations",
            "resolved",
        )
        for name in object_fields:
            if np.asarray(getattr(self, name)).shape != (n_objects,):
                raise ValueError(f"bank {name} must have shape {(n_objects,)}")
        row_index = np.asarray(self.row_index, dtype=np.int64)
        if len(np.unique(row_index)) != n_objects:
            raise ValueError("bank row_index values must be unique within a shard")
        methods = np.asarray(self.method, dtype=np.int8)
        if np.any(~np.isin(methods, tuple(POSTERIOR_METHOD_NAMES))):
            raise ValueError("bank contains an unknown posterior method code")
        counts = np.asarray(self.particle_count, dtype=np.int64)
        if np.any((counts <= 0) | (counts > capacity)):
            raise ValueError("bank particle_count is outside padded capacity")
        weights = np.asarray(self.normalized_weights, dtype=np.float64)
        for index, count in enumerate(counts):
            active = weights[index, :count]
            if not np.all(np.isfinite(active)) or np.any(active < 0.0):
                raise ValueError(
                    "bank active normalized weights must be finite nonnegative"
                )
            if not np.isclose(np.sum(active), 1.0, rtol=1.0e-6, atol=1.0e-8):
                raise ValueError("bank active normalized weights must sum to one")
            if np.any(weights[index, count:] != 0.0):
                raise ValueError("bank padded normalized weights must be zero")
        if self.features is None and not self.feature_reference:
            raise ValueError("bank must store features or a feature reference")
        if self.features is not None:
            features = np.asarray(self.features)
            if features.ndim != 2 or features.shape[0] != n_objects:
                raise ValueError("bank features must have shape [objects, features]")


def posterior_method_code(name: str) -> int:
    try:
        return POSTERIOR_METHOD_CODES[str(name)]
    except KeyError as error:
        raise ValueError(f"unknown posterior method: {name}") from error


def write_posterior_bank_shard(
    root: str | Path,
    shard_id: int,
    shard: PosteriorBankShard,
    provenance: PosteriorBankProvenance,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Atomically write one shard and its completion marker."""
    shard.validate()
    provenance.validate()
    root_path = Path(root)
    shards_dir = root_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    final = shards_dir / f"shard_{int(shard_id):05d}"
    if final.exists():
        if resume and is_posterior_bank_shard_complete(final, validate_arrays=True):
            validate_posterior_bank_shard_provenance(final, provenance)
            return read_shard_metadata(final)
        raise FileExistsError(
            f"posterior-bank shard already exists or is incomplete: {final}"
        )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".shard_{int(shard_id):05d}.",
            dir=shards_dir,
        )
    )
    try:
        arrays_path = temporary / "arrays.npz"
        arrays = _shard_arrays(shard)
        with arrays_path.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = {
            "schema_version": 1,
            "shard_id": int(shard_id),
            "object_count": shard.object_count,
            "particle_capacity": shard.particle_capacity,
            "latent_dim": shard.latent_dim,
            "row_index_min": int(np.min(shard.row_index)),
            "row_index_max": int(np.max(shard.row_index)),
            "method_counts": _method_counts(shard.method),
            "resolved_count": int(np.sum(np.asarray(shard.resolved, dtype=bool))),
            "arrays_sha256": sha256_file(arrays_path),
            "provenance": asdict(provenance),
            "feature_storage": (
                "inline" if shard.features is not None else "reference"
            ),
            "feature_reference": shard.feature_reference,
        }
        _write_json_fsync(temporary / "metadata.json", metadata)
        completion = {
            "status": "complete",
            "metadata_sha256": sha256_file(temporary / "metadata.json"),
            "arrays_sha256": metadata["arrays_sha256"],
        }
        _write_json_fsync(temporary / "COMPLETE.json", completion)
        os.replace(temporary, final)
        _fsync_directory(shards_dir)
        return metadata
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def is_posterior_bank_shard_complete(
    path: str | Path,
    *,
    validate_arrays: bool = False,
) -> bool:
    shard_dir = Path(path)
    arrays = shard_dir / "arrays.npz"
    metadata = shard_dir / "metadata.json"
    marker = shard_dir / "COMPLETE.json"
    if not (arrays.is_file() and metadata.is_file() and marker.is_file()):
        return False
    try:
        completion = json.loads(marker.read_text(encoding="utf-8"))
        if completion.get("status") != "complete":
            return False
        if completion.get("metadata_sha256") != sha256_file(metadata):
            return False
        if completion.get("arrays_sha256") != sha256_file(arrays):
            return False
        if validate_arrays:
            read_posterior_bank_shard(shard_dir).validate()
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def read_shard_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads((Path(path) / "metadata.json").read_text(encoding="utf-8"))


def validate_posterior_bank_shard_provenance(
    path: str | Path,
    provenance: PosteriorBankProvenance,
) -> None:
    """Reject resume when a complete shard belongs to another frozen snapshot."""
    provenance.validate()
    recorded = read_shard_metadata(path).get("provenance")
    if recorded != asdict(provenance):
        raise ValueError("posterior-bank resume provenance mismatch")


def validate_posterior_bank_manifest_provenance(
    manifest: dict[str, Any],
    *,
    expected_fields: dict[str, Any] | None = None,
) -> PosteriorBankProvenance:
    """Validate a merged bank contract and any phase-specific expected fields."""
    recorded = manifest.get("provenance")
    if not isinstance(recorded, dict):
        raise ValueError("posterior-bank manifest lacks provenance")
    try:
        provenance = PosteriorBankProvenance(**recorded)
    except TypeError as error:
        raise ValueError(
            "posterior-bank manifest provenance schema mismatch"
        ) from error
    provenance.validate()
    for name, expected in (expected_fields or {}).items():
        if recorded.get(name) != expected:
            raise ValueError(f"posterior-bank manifest provenance mismatch: {name}")
    return provenance


def read_posterior_bank_shard(path: str | Path) -> PosteriorBankShard:
    shard_dir = Path(path)
    metadata = read_shard_metadata(shard_dir)
    with np.load(shard_dir / "arrays.npz", allow_pickle=False) as arrays:
        payload = {name: np.asarray(arrays[name]) for name in arrays.files}
    features = payload.pop("features", None)
    return PosteriorBankShard(
        **payload,
        features=features,
        feature_reference=metadata.get("feature_reference"),
    )


def merge_posterior_bank_shards(
    bank_root: str | Path,
    shard_paths: list[str | Path],
    *,
    expected_row_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate and index shards without loading their particle arrays together."""
    root = Path(bank_root)
    if not shard_paths:
        raise ValueError("posterior-bank merge requires at least one shard")
    records = []
    all_rows: list[np.ndarray] = []
    canonical_provenance = None
    for raw_path in sorted(Path(value).resolve() for value in shard_paths):
        if not is_posterior_bank_shard_complete(raw_path):
            raise ValueError(f"posterior-bank shard is incomplete: {raw_path}")
        metadata = read_shard_metadata(raw_path)
        provenance = metadata["provenance"]
        if canonical_provenance is None:
            canonical_provenance = provenance
        elif provenance != canonical_provenance:
            raise ValueError("posterior-bank shard provenance mismatch")
        with np.load(raw_path / "arrays.npz", allow_pickle=False) as arrays:
            rows = np.asarray(arrays["row_index"], dtype=np.int64)
        all_rows.append(rows)
        records.append(
            {
                "path": str(raw_path),
                "shard_id": int(metadata["shard_id"]),
                "object_count": int(metadata["object_count"]),
                "row_index_min": int(metadata["row_index_min"]),
                "row_index_max": int(metadata["row_index_max"]),
                "arrays_sha256": metadata["arrays_sha256"],
                "method_counts": metadata["method_counts"],
                "resolved_count": int(metadata["resolved_count"]),
            }
        )
    rows = np.concatenate(all_rows)
    if len(np.unique(rows)) != len(rows):
        raise ValueError("posterior-bank shards contain duplicate row indices")
    if expected_row_indices is not None:
        expected = np.asarray(expected_row_indices, dtype=np.int64)
        if not np.array_equal(np.sort(rows), np.sort(expected)):
            raise ValueError(
                "posterior-bank merged rows do not match expected catalogue"
            )
    manifest = {
        "status": "complete",
        "schema_version": 1,
        "object_count": int(len(rows)),
        "shard_count": len(records),
        "shards": records,
        "provenance": canonical_provenance,
        "merge_contract": "streaming shard index; particles are not concatenated in host memory",
    }
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "posterior_bank_manifest.json", manifest)
    return manifest


def iter_posterior_bank_shards(
    manifest_path: str | Path,
) -> Iterator[PosteriorBankShard]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for record in manifest["shards"]:
        yield read_posterior_bank_shard(record["path"])


def reweight_posterior_particles(
    normalized_weights: np.ndarray,
    source_logprior: np.ndarray,
    new_logprior: np.ndarray,
    particle_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the exact prior-density ratio and return weights, ESS and max weight."""
    weights = np.asarray(normalized_weights, dtype=np.float64)
    source = np.asarray(source_logprior, dtype=np.float64)
    candidate = np.asarray(new_logprior, dtype=np.float64)
    counts = np.asarray(particle_count, dtype=np.int64)
    if weights.shape != source.shape or weights.shape != candidate.shape:
        raise ValueError("bank reweight arrays must have identical shapes")
    if counts.shape != (weights.shape[0],):
        raise ValueError("bank particle_count shape mismatch")
    result = np.zeros_like(weights)
    ess = np.zeros(weights.shape[0], dtype=np.float64)
    maximum = np.zeros(weights.shape[0], dtype=np.float64)
    for index, count in enumerate(counts):
        active = weights[index, :count]
        log_weights = np.where(active > 0.0, np.log(active), -np.inf)
        log_weights = log_weights + candidate[index, :count] - source[index, :count]
        finite = np.isfinite(log_weights)
        if not np.any(finite):
            raise ValueError(f"bank reweight has no finite particle for object {index}")
        offset = np.max(log_weights[finite])
        unnormalized = np.where(finite, np.exp(log_weights - offset), 0.0)
        normalized = unnormalized / np.sum(unnormalized)
        result[index, :count] = normalized
        ess[index] = 1.0 / np.sum(np.square(normalized))
        maximum[index] = np.max(normalized)
    return result, ess, maximum


def reweight_posterior_bank_shard(
    shard: PosteriorBankShard,
    new_logprior: np.ndarray,
) -> PosteriorBankShard:
    weights, ess, maximum = reweight_posterior_particles(
        shard.normalized_weights,
        shard.source_logprior,
        new_logprior,
        shard.particle_count,
    )
    return replace(
        shard,
        normalized_weights=weights,
        source_logprior=np.asarray(new_logprior, dtype=np.float64),
        ess=ess,
        max_weight=maximum,
    )


def low_reweight_ess_rows(
    row_index: np.ndarray,
    ess: np.ndarray,
    particle_count: np.ndarray,
    *,
    minimum_ess_fraction: float,
) -> np.ndarray:
    if not 0.0 < float(minimum_ess_fraction) <= 1.0:
        raise ValueError("minimum_ess_fraction must be in (0, 1]")
    rows = np.asarray(row_index, dtype=np.int64)
    fraction = np.asarray(ess, dtype=float) / np.asarray(particle_count, dtype=float)
    return rows[~np.isfinite(fraction) | (fraction < float(minimum_ess_fraction))]


def replace_posterior_bank_rows(
    base: PosteriorBankShard,
    replacement: PosteriorBankShard,
) -> PosteriorBankShard:
    """Replace an exact row subset while preserving the base object order."""
    base.validate()
    replacement.validate()
    if base.particle_capacity != replacement.particle_capacity:
        raise ValueError("posterior-bank replacement particle capacity mismatch")
    if base.latent_dim != replacement.latent_dim:
        raise ValueError("posterior-bank replacement latent dimension mismatch")
    lookup = {
        int(row): index
        for index, row in enumerate(np.asarray(base.row_index, dtype=np.int64))
    }
    replacement_rows = np.asarray(replacement.row_index, dtype=np.int64)
    if any(int(row) not in lookup for row in replacement_rows):
        raise ValueError("posterior-bank replacement contains rows outside base shard")
    destination = np.asarray([lookup[int(row)] for row in replacement_rows])
    values: dict[str, Any] = {}
    array_fields = tuple(
        name
        for name in PosteriorBankShard.__dataclass_fields__
        if name not in {"features", "feature_reference"}
    )
    for name in array_fields:
        current = np.asarray(getattr(base, name)).copy()
        current[destination] = np.asarray(getattr(replacement, name))
        values[name] = current
    if base.features is None:
        if replacement.features is not None:
            raise ValueError("cannot insert inline features into reference-only bank")
        features = None
        feature_reference = base.feature_reference
    else:
        if replacement.features is None:
            raise ValueError("replacement bank omitted required inline features")
        features = np.asarray(base.features).copy()
        features[destination] = np.asarray(replacement.features)
        feature_reference = base.feature_reference
    result = PosteriorBankShard(
        **values,
        features=features,
        feature_reference=feature_reference,
    )
    result.validate()
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shard_arrays(shard: PosteriorBankShard) -> dict[str, np.ndarray]:
    arrays = {
        "row_index": np.asarray(shard.row_index, dtype=np.int64),
        "object_id": np.asarray(shard.object_id, dtype=str),
        "method": np.asarray(shard.method, dtype=np.int8),
        "particles": np.asarray(shard.particles, dtype=np.float32),
        "normalized_weights": np.asarray(shard.normalized_weights, dtype=np.float64),
        "source_logprior": np.asarray(shard.source_logprior, dtype=np.float64),
        "particle_count": np.asarray(shard.particle_count, dtype=np.int16),
        "ess": np.asarray(shard.ess, dtype=np.float64),
        "max_weight": np.asarray(shard.max_weight, dtype=np.float64),
        "beta_final": np.asarray(shard.beta_final, dtype=np.float32),
        "logz": np.asarray(shard.logz, dtype=np.float64),
        "stage_count": np.asarray(shard.stage_count, dtype=np.int16),
        "acceptance": np.asarray(shard.acceptance, dtype=np.float32),
        "ancestor_ess": np.asarray(shard.ancestor_ess, dtype=np.float32),
        "unique_ancestor_fraction": np.asarray(
            shard.unique_ancestor_fraction, dtype=np.float32
        ),
        "movement_squared": np.asarray(shard.movement_squared, dtype=np.float32),
        "moved_particle_fraction": np.asarray(
            shard.moved_particle_fraction, dtype=np.float32
        ),
        "dsps_evaluations": np.asarray(shard.dsps_evaluations, dtype=np.int32),
        "resolved": np.asarray(shard.resolved, dtype=bool),
    }
    if shard.features is not None:
        arrays["features"] = np.asarray(shard.features, dtype=np.float32)
    return arrays


def _method_counts(method: np.ndarray) -> dict[str, int]:
    values = np.asarray(method, dtype=np.int8)
    return {
        POSTERIOR_METHOD_NAMES[code]: int(np.sum(values == code))
        for code in sorted(POSTERIOR_METHOD_NAMES)
    }


def _is_sha256(value: str) -> bool:
    return len(str(value)) == 64 and all(
        character in "0123456789abcdef" for character in str(value).lower()
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_json_fsync(temporary, payload)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _write_json_fsync(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
