"""Low-cost selection-corrected variational EM primitives.

The E-step banks contain object-major joint draws from ``q_psi(x | y)``.  They
are deliberately proposal-only: no point estimator and no importance weight is
introduced.  The M-step targets the parent population prior with

    -E_q[log p_eta(x)] + log E_{p_eta}[beta(x)].

The selection normalization is evaluated from a fixed reference bank drawn
from the frozen source prior.  This keeps DSPS outside the optimization loop
while retaining a differentiable importance-sampling estimate of ``alpha``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import require_amortized_dependencies

eqx, _optax = require_amortized_dependencies()


@dataclass(frozen=True)
class ArrayShardContract:
    """Immutable provenance shared by every shard in one array bank."""

    kind: str
    dataset_sha256: str
    checkpoint_sha256: str
    latent_transform_sha256: str
    code_commit: str
    truth_used: bool
    draws_per_object: int | None = None
    feature_stats_sha256: str | None = None
    row_indices_sha256: str | None = None
    selection_event: str | None = None

    def validate(self) -> None:
        if self.kind not in {
            "q_train",
            "q_validation",
            "q_evaluation",
            "selection_reference",
            "selection_audit",
            "prior_evaluation",
        }:
            raise ValueError(f"unknown population-VEM bank kind: {self.kind}")
        for name in (
            "dataset_sha256",
            "checkpoint_sha256",
            "latent_transform_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA256 digest")
        for name in ("feature_stats_sha256", "row_indices_sha256"):
            value = getattr(self, name)
            if value is not None and not _is_sha256(value):
                raise ValueError(f"{name} must be a SHA256 digest when present")
        if not self.code_commit:
            raise ValueError("population-VEM bank requires a code commit")
        if self.kind != "selection_audit" and self.truth_used:
            raise ValueError("truth is forbidden outside the isolated selection audit")
        if self.kind.startswith("q_"):
            if self.draws_per_object is None or int(self.draws_per_object) <= 0:
                raise ValueError("q banks require a positive draws_per_object")
            if self.feature_stats_sha256 is None or self.row_indices_sha256 is None:
                raise ValueError("q banks require feature and row-index provenance")
        if (
            self.kind
            in {
                "selection_reference",
                "selection_audit",
                "prior_evaluation",
            }
            and not self.selection_event
        ):
            raise ValueError(
                "selection-bearing banks require an explicit selection event"
            )


class FixedReferenceSelectionTerms(NamedTuple):
    log_alpha: jnp.ndarray
    alpha: jnp.ndarray
    ess: jnp.ndarray
    ess_fraction: jnp.ndarray
    relative_mc_error: jnp.ndarray
    maximum_normalized_weight: jnp.ndarray
    finite: jnp.ndarray


class FixedReferencePriorMetrics(NamedTuple):
    loss: jnp.ndarray
    data_nll: jnp.ndarray
    log_alpha: jnp.ndarray
    alpha: jnp.ndarray
    reference_ess_fraction: jnp.ndarray
    alpha_relative_mc_error: jnp.ndarray
    maximum_reference_weight: jnp.ndarray
    source_to_candidate_kl: jnp.ndarray
    proposed_source_to_candidate_kl: jnp.ndarray
    proposed_reference_ess_fraction: jnp.ndarray
    raw_gradient_norm: jnp.ndarray
    gradients_finite: jnp.ndarray
    valid_objects: jnp.ndarray
    update_applied: jnp.ndarray
    rejection_code: jnp.ndarray


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_directory(repo: Path) -> Path:
    marker = repo / ".git"
    if marker.is_dir():
        return marker.resolve()
    if marker.is_file():
        value = marker.read_text(encoding="utf-8").strip()
        prefix = "gitdir:"
        if not value.lower().startswith(prefix):
            raise ValueError(f"invalid Git worktree marker: {marker}")
        directory = Path(value[len(prefix) :].strip())
        if not directory.is_absolute():
            directory = marker.parent / directory
        return directory.resolve()
    raise FileNotFoundError(f"missing Git metadata: {marker}")


def _read_git_head(repo: str | Path) -> str:
    """Read HEAD from a repository or linked worktree without invoking Git."""
    directory = _git_directory(Path(repo).resolve())
    head = (directory / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        actual = head
    else:
        reference = head.removeprefix("ref: ").strip()
        common = directory
        common_marker = directory / "commondir"
        if common_marker.is_file():
            common = (directory / common_marker.read_text().strip()).resolve()
        loose = common / reference
        if loose.is_file():
            actual = loose.read_text(encoding="utf-8").strip()
        else:
            actual = ""
            packed = common / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.startswith(("#", "^")):
                        continue
                    fields = line.split()
                    if len(fields) == 2 and fields[1] == reference:
                        actual = fields[0]
                        break
            if not actual:
                raise ValueError(f"cannot resolve Git reference {reference!r}")
    if len(actual) not in {40, 64} or any(
        character not in "0123456789abcdefABCDEF" for character in actual
    ):
        raise ValueError(f"invalid Git HEAD value: {actual!r}")
    return actual.lower()


def require_git_commit(repo: str | Path, expected: str) -> str:
    """Fail if a queued workflow is no longer using its frozen code commit."""
    actual = _read_git_head(repo)
    if actual != str(expected):
        raise ValueError(
            "population-VEM code provenance mismatch: "
            f"expected={expected}, actual={actual}"
        )
    return actual


def resolve_manifest_config(
    manifest: dict[str, Any], section: str, repo: str | Path
) -> Path:
    """Resolve and verify a config from the immutable code snapshot."""
    record = manifest[section]
    relative = record.get("repo_relative_path")
    if not relative:
        raise ValueError(f"manifest {section} is missing repo_relative_path")
    path = (Path(repo) / str(relative)).resolve()
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"population-VEM {section} SHA256 mismatch: {path}")
    return path


def write_array_bank_shard(
    root: str | Path,
    shard_id: int,
    arrays: dict[str, np.ndarray],
    contract: ArrayShardContract,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Atomically publish one compressed array shard and its receipt."""
    contract.validate()
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    _validate_arrays_for_kind(contract, normalized)
    root_path = Path(root)
    shards = root_path / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    final = shards / f"shard_{int(shard_id):05d}"
    if final.exists():
        if resume and is_array_bank_shard_complete(final, validate_arrays=True):
            receipt = read_array_bank_shard_receipt(final)
            if receipt.get("contract") != asdict(contract):
                raise ValueError("population-VEM shard resume provenance mismatch")
            return receipt
        raise FileExistsError(f"array bank shard exists or is incomplete: {final}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".shard_{int(shard_id):05d}.", dir=shards)
    )
    try:
        array_path = temporary / "arrays.npz"
        with array_path.open("wb") as stream:
            np.savez_compressed(stream, **normalized)
            stream.flush()
            os.fsync(stream.fileno())
        receipt = {
            "status": "complete",
            "schema_version": 1,
            "shard_id": int(shard_id),
            "contract": asdict(contract),
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in sorted(normalized.items())
            },
            "arrays_sha256": sha256_file(array_path),
        }
        _atomic_json(temporary / "receipt.json", receipt)
        _atomic_json(
            temporary / "COMPLETE.json",
            {
                "status": "complete",
                "receipt_sha256": sha256_file(temporary / "receipt.json"),
                "arrays_sha256": receipt["arrays_sha256"],
            },
        )
        os.replace(temporary, final)
        _fsync_directory(shards)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def is_array_bank_shard_complete(
    path: str | Path,
    *,
    validate_arrays: bool = False,
) -> bool:
    root = Path(path)
    arrays_path = root / "arrays.npz"
    receipt_path = root / "receipt.json"
    marker_path = root / "COMPLETE.json"
    if not (arrays_path.is_file() and receipt_path.is_file() and marker_path.is_file()):
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete" or marker.get("status") != "complete":
            return False
        if marker.get("receipt_sha256") != sha256_file(receipt_path):
            return False
        if marker.get("arrays_sha256") != sha256_file(arrays_path):
            return False
        contract = ArrayShardContract(**receipt["contract"])
        contract.validate()
        if validate_arrays:
            arrays = read_array_bank_shard(root)
            _validate_arrays_for_kind(contract, arrays)
            recorded = receipt.get("arrays", {})
            actual = {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in sorted(arrays.items())
            }
            if actual != recorded:
                return False
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def read_array_bank_shard(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path) / "arrays.npz", allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def read_array_bank_shard_receipt(path: str | Path) -> dict[str, Any]:
    return json.loads((Path(path) / "receipt.json").read_text(encoding="utf-8"))


def merge_array_bank_shards(
    root: str | Path,
    *,
    expected_shards: int,
    expected_row_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate a bank and write a streaming manifest without concatenating draws."""
    bank = Path(root)
    if int(expected_shards) <= 0:
        raise ValueError("expected_shards must be positive")
    records: list[dict[str, Any]] = []
    canonical_contract = None
    row_parts: list[np.ndarray] = []
    for shard_id in range(int(expected_shards)):
        path = bank / "shards" / f"shard_{shard_id:05d}"
        if not is_array_bank_shard_complete(path, validate_arrays=True):
            raise ValueError(f"incomplete population-VEM shard: {path}")
        receipt = read_array_bank_shard_receipt(path)
        contract = receipt["contract"]
        if canonical_contract is None:
            canonical_contract = contract
        elif contract != canonical_contract:
            raise ValueError("population-VEM bank shards have different provenance")
        arrays = read_array_bank_shard(path)
        if "row_index" in arrays:
            row_parts.append(np.asarray(arrays["row_index"], dtype=np.int64))
        records.append(
            {
                "shard_id": shard_id,
                "path": str(path.resolve()),
                "arrays_sha256": receipt["arrays_sha256"],
                "arrays": receipt["arrays"],
            }
        )
    if expected_row_indices is not None:
        if not row_parts:
            raise ValueError("row-index validation requested for a row-free bank")
        rows = np.concatenate(row_parts)
        expected = np.asarray(expected_row_indices, dtype=np.int64)
        if len(np.unique(rows)) != len(rows):
            raise ValueError("population-VEM bank contains duplicate row indices")
        if not np.array_equal(np.sort(rows), np.sort(expected)):
            raise ValueError("population-VEM bank rows differ from the expected cohort")
    manifest = {
        "status": "complete",
        "schema_version": 1,
        "contract": canonical_contract,
        "shard_count": len(records),
        "shards": records,
        "streaming": True,
    }
    _atomic_json(bank / "bank_manifest.json", manifest)
    return manifest


def iter_array_bank_shards(
    manifest_path: str | Path,
) -> Iterable[dict[str, np.ndarray]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for record in manifest["shards"]:
        yield read_array_bank_shard(record["path"])


def fixed_reference_selection_terms(
    candidate_log_prob: jnp.ndarray,
    reference_log_prob: jnp.ndarray,
    reference_log_beta: jnp.ndarray,
) -> FixedReferenceSelectionTerms:
    """Estimate alpha and reference support from fixed importance samples."""
    dtype = jnp.result_type(
        candidate_log_prob,
        reference_log_prob,
        reference_log_beta,
    )
    candidate = jnp.asarray(candidate_log_prob, dtype=dtype)
    reference = jnp.asarray(reference_log_prob, dtype=dtype)
    log_beta = jnp.asarray(reference_log_beta, dtype=dtype)
    if candidate.shape != reference.shape or candidate.shape != log_beta.shape:
        raise ValueError("fixed-reference alpha arrays must have identical shapes")
    log_weight = log_beta + candidate - reference
    finite_weight = jnp.isfinite(log_weight)
    safe_log_weight = jnp.where(finite_weight, log_weight, -jnp.inf)
    maximum = jnp.max(safe_log_weight)
    stable = jnp.where(finite_weight, jnp.exp(safe_log_weight - maximum), 0.0)
    count = jnp.asarray(candidate.size, dtype=candidate.dtype)
    weight_sum = jnp.sum(stable)
    weight_sum_square = jnp.sum(jnp.square(stable))
    tiny = jnp.asarray(jnp.finfo(candidate.dtype).tiny, dtype=candidate.dtype)
    safe_sum = jnp.maximum(weight_sum, tiny)
    log_alpha = maximum + jnp.log(safe_sum / count)
    alpha = jnp.exp(log_alpha)
    ess = weight_sum**2 / jnp.maximum(weight_sum_square, tiny)
    mean = weight_sum / count
    second_moment = weight_sum_square / count
    variance = jnp.maximum(second_moment - mean**2, 0.0)
    relative_error = jnp.sqrt(variance / count) / jnp.maximum(mean, tiny)
    normalized_maximum = jnp.max(stable) / safe_sum
    valid = jnp.any(finite_weight) & jnp.isfinite(log_alpha) & (weight_sum > 0.0)
    return FixedReferenceSelectionTerms(
        log_alpha=log_alpha,
        alpha=alpha,
        ess=ess,
        ess_fraction=ess / count,
        relative_mc_error=relative_error,
        maximum_normalized_weight=normalized_maximum,
        finite=valid,
    )


def selection_calibration_summary(
    beta: np.ndarray,
    selected: np.ndarray,
    redshift: np.ndarray,
    *,
    probability_bins: int = 10,
    redshift_bins: int = 10,
    minimum_redshift_bin_objects: int = 100,
    maximum_global_error: float = 0.03,
    maximum_ece: float = 0.05,
    maximum_redshift_bin_error: float = 0.10,
) -> dict[str, Any]:
    """Calibrate beta against frozen closure labels without exposing truth to VEM."""
    probability = np.asarray(beta, dtype=np.float64).reshape(-1)
    outcome = np.asarray(selected, dtype=bool).reshape(-1)
    z = np.asarray(redshift, dtype=np.float64).reshape(-1)
    if not (len(probability) == len(outcome) == len(z)) or len(probability) == 0:
        raise ValueError("selection calibration arrays must have equal nonzero length")
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("beta probabilities must be finite and lie in [0, 1]")
    if not np.all(np.isfinite(z)):
        raise ValueError("selection calibration redshift must be finite")
    global_error = float(abs(np.mean(probability) - np.mean(outcome)))
    probability_edges = np.linspace(0.0, 1.0, int(probability_bins) + 1)
    calibration_rows = []
    ece = 0.0
    for index in range(int(probability_bins)):
        mask = (probability >= probability_edges[index]) & (
            probability < probability_edges[index + 1]
        )
        if index == int(probability_bins) - 1:
            mask |= probability == 1.0
        if not np.any(mask):
            continue
        predicted = float(np.mean(probability[mask]))
        observed = float(np.mean(outcome[mask]))
        fraction = float(np.mean(mask))
        ece += fraction * abs(predicted - observed)
        calibration_rows.append(
            {
                "bin": index,
                "objects": int(np.sum(mask)),
                "predicted": predicted,
                "observed": observed,
                "absolute_error": abs(predicted - observed),
            }
        )
    z_edges = np.unique(np.quantile(z, np.linspace(0.0, 1.0, int(redshift_bins) + 1)))
    redshift_rows = []
    redshift_errors = []
    for index in range(max(len(z_edges) - 1, 0)):
        mask = (z >= z_edges[index]) & (z < z_edges[index + 1])
        if index == len(z_edges) - 2:
            mask |= z == z_edges[index + 1]
        objects = int(np.sum(mask))
        if objects < int(minimum_redshift_bin_objects):
            continue
        predicted = float(np.mean(probability[mask]))
        observed = float(np.mean(outcome[mask]))
        error = abs(predicted - observed)
        redshift_errors.append(error)
        redshift_rows.append(
            {
                "bin": index,
                "z_min": float(z_edges[index]),
                "z_max": float(z_edges[index + 1]),
                "objects": objects,
                "predicted": predicted,
                "observed": observed,
                "absolute_error": error,
            }
        )
    maximum_z_error = float(max(redshift_errors, default=np.inf))
    passed = bool(
        global_error <= float(maximum_global_error)
        and ece <= float(maximum_ece)
        and maximum_z_error <= float(maximum_redshift_bin_error)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "objects": len(probability),
        "mean_beta": float(np.mean(probability)),
        "observed_selected_fraction": float(np.mean(outcome)),
        "global_absolute_error": global_error,
        "brier_score": float(np.mean(np.square(probability - outcome))),
        "expected_calibration_error": float(ece),
        "maximum_redshift_bin_error": maximum_z_error,
        "thresholds": {
            "maximum_global_error": float(maximum_global_error),
            "maximum_ece": float(maximum_ece),
            "maximum_redshift_bin_error": float(maximum_redshift_bin_error),
            "minimum_redshift_bin_objects": int(minimum_redshift_bin_objects),
        },
        "probability_bins": calibration_rows,
        "redshift_bins": redshift_rows,
        "truth_role": "isolated frozen closure audit only; never a training input",
    }


def make_pmap_fixed_reference_prior_step(
    *,
    optimizer: Any,
    minimum_reference_ess_fraction: float,
    maximum_alpha_relative_mc_error: float,
    maximum_kl_per_dimension: float,
):
    """Build a four-way-capable prior-only update over fixed q/reference banks."""
    if not 0.0 < float(minimum_reference_ess_fraction) <= 1.0:
        raise ValueError("minimum_reference_ess_fraction must lie in (0, 1]")
    if float(maximum_alpha_relative_mc_error) <= 0.0:
        raise ValueError("maximum_alpha_relative_mc_error must be positive")
    if float(maximum_kl_per_dimension) <= 0.0:
        raise ValueError("maximum_kl_per_dimension must be positive")
    array_axis = eqx.if_array(0)

    @eqx.filter_pmap(
        axis_name="devices",
        in_axes=(
            array_axis,
            array_axis,
            array_axis,
            array_axis,
            array_axis,
            array_axis,
            array_axis,
            array_axis,
            array_axis,
        ),
        out_axes=(array_axis, array_axis, array_axis),
    )
    def step(
        prior,
        source_prior,
        optimizer_state,
        q_x,
        q_valid,
        reference_x,
        reference_log_prob,
        reference_log_beta,
        trust_strength,
    ):
        q_x = jax.lax.stop_gradient(q_x)
        q_valid = jax.lax.stop_gradient(q_valid)
        reference_x = jax.lax.stop_gradient(reference_x)
        reference_log_prob = jax.lax.stop_gradient(reference_log_prob)
        reference_log_beta = jax.lax.stop_gradient(reference_log_beta)

        def objective(candidate_prior):
            q_log_prob = candidate_prior.log_prob(q_x)
            q_finite = jnp.all(jnp.isfinite(q_log_prob), axis=1)
            usable = q_valid & q_finite
            per_object = -jnp.mean(
                jnp.where(jnp.isfinite(q_log_prob), q_log_prob, 0.0), axis=1
            )
            local_data_sum = jnp.sum(jnp.where(usable, per_object, 0.0))
            local_objects = jnp.sum(usable.astype(per_object.dtype))
            global_data_sum = jax.lax.psum(local_data_sum, "devices")
            global_objects = jax.lax.psum(local_objects, "devices")
            data_nll = global_data_sum / jnp.maximum(global_objects, 1.0)

            candidate_reference_log_prob = candidate_prior.log_prob(reference_x)
            selection = _distributed_fixed_reference_terms(
                candidate_reference_log_prob,
                reference_log_prob,
                reference_log_beta,
            )
            source_log_prob = jax.lax.stop_gradient(source_prior.log_prob(reference_x))
            local_kl_sum = jnp.sum(source_log_prob - candidate_reference_log_prob)
            local_reference_count = jnp.asarray(
                reference_x.shape[0], dtype=local_kl_sum.dtype
            )
            global_kl_sum = jax.lax.psum(local_kl_sum, "devices")
            global_reference_count = jax.lax.psum(local_reference_count, "devices")
            trust_kl = global_kl_sum / global_reference_count
            loss = data_nll + selection.log_alpha + trust_strength * trust_kl
            return loss, (data_nll, selection, trust_kl, global_objects)

        (loss, auxiliary), gradients = eqx.filter_value_and_grad(
            objective, has_aux=True
        )(prior)
        gradients = _pmean_tree(gradients, "devices")
        raw_gradient_norm = _tree_l2_norm(gradients)
        gradients_finite = _tree_all_finite(gradients)
        data_nll, selection, trust_kl, valid_objects = auxiliary
        pre_update_ok = jnp.isfinite(loss) & gradients_finite
        pre_update_ok &= valid_objects > 0.0
        pre_update_ok &= selection.finite
        pre_update_ok &= selection.relative_mc_error <= float(
            maximum_alpha_relative_mc_error
        )
        pre_update_ok &= selection.ess_fraction >= float(minimum_reference_ess_fraction)
        safe_gradients = jax.tree_util.tree_map(
            lambda value: (
                jnp.where(pre_update_ok, value, jnp.zeros_like(value))
                if value is not None
                else None
            ),
            gradients,
        )
        updates, proposed_optimizer_state = optimizer.update(
            safe_gradients,
            optimizer_state,
            eqx.filter(prior, eqx.is_inexact_array),
        )
        proposed_prior = eqx.apply_updates(prior, updates)
        proposed_log_prob = proposed_prior.log_prob(reference_x)
        proposed_selection = _distributed_fixed_reference_terms(
            proposed_log_prob,
            reference_log_prob,
            reference_log_beta,
        )
        source_log_prob = jax.lax.stop_gradient(source_prior.log_prob(reference_x))
        proposed_kl = jax.lax.psum(
            jnp.sum(source_log_prob - proposed_log_prob), "devices"
        ) / jax.lax.psum(
            jnp.asarray(reference_x.shape[0], dtype=proposed_log_prob.dtype),
            "devices",
        )
        dimension = int(prior.latent_dim)
        proposed_ok = jnp.isfinite(proposed_kl)
        proposed_ok &= jnp.abs(proposed_kl) <= (
            float(maximum_kl_per_dimension) * dimension
        )
        proposed_ok &= proposed_selection.finite
        proposed_ok &= proposed_selection.ess_fraction >= float(
            minimum_reference_ess_fraction
        )
        apply_update = jax.lax.pmin(
            (pre_update_ok & proposed_ok).astype(jnp.int32), "devices"
        ).astype(jnp.bool_)
        prior = _select_tree(proposed_prior, prior, apply_update)
        optimizer_state = _select_tree(
            proposed_optimizer_state, optimizer_state, apply_update
        )
        rejection_code = jnp.where(
            ~jnp.isfinite(loss) | ~gradients_finite,
            1,
            jnp.where(
                valid_objects <= 0.0,
                2,
                jnp.where(
                    ~selection.finite,
                    3,
                    jnp.where(
                        selection.relative_mc_error
                        > float(maximum_alpha_relative_mc_error),
                        4,
                        jnp.where(
                            selection.ess_fraction
                            < float(minimum_reference_ess_fraction),
                            5,
                            jnp.where(~proposed_ok, 6, 0),
                        ),
                    ),
                ),
            ),
        )
        metrics = FixedReferencePriorMetrics(
            loss=loss,
            data_nll=data_nll,
            log_alpha=selection.log_alpha,
            alpha=selection.alpha,
            reference_ess_fraction=selection.ess_fraction,
            alpha_relative_mc_error=selection.relative_mc_error,
            maximum_reference_weight=selection.maximum_normalized_weight,
            source_to_candidate_kl=trust_kl,
            proposed_source_to_candidate_kl=proposed_kl,
            proposed_reference_ess_fraction=proposed_selection.ess_fraction,
            raw_gradient_norm=raw_gradient_norm,
            gradients_finite=gradients_finite,
            valid_objects=valid_objects,
            update_applied=apply_update,
            rejection_code=jnp.asarray(rejection_code, dtype=jnp.int32),
        )
        return prior, optimizer_state, metrics

    return step


def _distributed_fixed_reference_terms(
    candidate_log_prob: jnp.ndarray,
    reference_log_prob: jnp.ndarray,
    reference_log_beta: jnp.ndarray,
) -> FixedReferenceSelectionTerms:
    dtype = jnp.result_type(
        candidate_log_prob,
        reference_log_prob,
        reference_log_beta,
    )
    candidate_log_prob = jnp.asarray(candidate_log_prob, dtype=dtype)
    log_weight = (
        jnp.asarray(reference_log_beta, dtype=dtype)
        + candidate_log_prob
        - jnp.asarray(reference_log_prob, dtype=dtype)
    )
    finite = jnp.isfinite(log_weight)
    local_maximum = jnp.max(jnp.where(finite, log_weight, -jnp.inf))
    global_maximum = jax.lax.pmax(jax.lax.stop_gradient(local_maximum), "devices")
    stable = jnp.where(finite, jnp.exp(log_weight - global_maximum), 0.0)
    weight_sum = jax.lax.psum(jnp.sum(stable), "devices")
    weight_sum_square = jax.lax.psum(jnp.sum(jnp.square(stable)), "devices")
    count = jax.lax.psum(jnp.asarray(candidate_log_prob.size, dtype=dtype), "devices")
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    safe_sum = jnp.maximum(weight_sum, tiny)
    log_alpha = global_maximum + jnp.log(safe_sum / count)
    alpha = jnp.exp(log_alpha)
    ess = weight_sum**2 / jnp.maximum(weight_sum_square, tiny)
    mean = weight_sum / count
    second_moment = weight_sum_square / count
    variance = jnp.maximum(second_moment - mean**2, 0.0)
    relative_error = jnp.sqrt(variance / count) / jnp.maximum(mean, tiny)
    local_maximum_weight = jnp.max(stable) / safe_sum
    maximum_weight = jax.lax.pmax(
        jax.lax.stop_gradient(local_maximum_weight), "devices"
    )
    valid_count = jax.lax.psum(jnp.sum(finite.astype(jnp.int32)), "devices")
    return FixedReferenceSelectionTerms(
        log_alpha=log_alpha,
        alpha=alpha,
        ess=ess,
        ess_fraction=ess / count,
        relative_mc_error=relative_error,
        maximum_normalized_weight=maximum_weight,
        finite=(valid_count > 0) & jnp.isfinite(log_alpha) & (weight_sum > 0.0),
    )


def _validate_arrays_for_kind(
    contract: ArrayShardContract,
    arrays: dict[str, np.ndarray],
) -> None:
    if not arrays:
        raise ValueError("array bank shard cannot be empty")
    if contract.kind.startswith("q_"):
        required = {"row_index", "x", "log_q"}
        missing = required - set(arrays)
        if missing:
            raise ValueError(f"q bank shard is missing arrays: {sorted(missing)}")
        rows = np.asarray(arrays["row_index"], dtype=np.int64)
        x = np.asarray(arrays["x"])
        log_q = np.asarray(arrays["log_q"])
        if rows.ndim != 1 or len(np.unique(rows)) != len(rows):
            raise ValueError("q bank row_index must be one-dimensional and unique")
        expected_draws = int(contract.draws_per_object or 0)
        if x.ndim != 3 or x.shape[:2] != (len(rows), expected_draws):
            raise ValueError("q bank x must have shape [objects, draws, latent]")
        if log_q.shape != (len(rows), expected_draws):
            raise ValueError("q bank log_q must have shape [objects, draws]")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(log_q)):
            raise ValueError("q bank contains non-finite draws or log densities")
    elif contract.kind == "selection_reference":
        required = {"x", "log_p_reference", "log_beta"}
        missing = required - set(arrays)
        if missing:
            raise ValueError(
                f"selection reference shard is missing arrays: {sorted(missing)}"
            )
        x = np.asarray(arrays["x"])
        log_p = np.asarray(arrays["log_p_reference"])
        log_beta = np.asarray(arrays["log_beta"])
        if x.ndim != 2 or log_p.shape != x.shape[:1] or log_beta.shape != x.shape[:1]:
            raise ValueError("selection reference arrays have incompatible shapes")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(log_p)):
            raise ValueError("selection reference contains non-finite prior draws")
        if np.any(np.isnan(log_beta)) or np.any(log_beta > 0.0):
            raise ValueError("selection reference log_beta must be <= 0 and not NaN")
    elif contract.kind == "selection_audit":
        required = {"row_index", "beta", "selected", "redshift"}
        missing = required - set(arrays)
        if missing:
            raise ValueError(f"selection audit shard is missing: {sorted(missing)}")
        length = len(np.asarray(arrays["row_index"]))
        if any(np.asarray(arrays[name]).shape != (length,) for name in required):
            raise ValueError("selection audit arrays must all be one-dimensional")
        rows = np.asarray(arrays["row_index"], dtype=np.int64)
        beta = np.asarray(arrays["beta"], dtype=np.float64)
        redshift = np.asarray(arrays["redshift"], dtype=np.float64)
        if len(np.unique(rows)) != len(rows):
            raise ValueError("selection audit row indices must be unique")
        if not np.all(np.isfinite(beta)) or np.any((beta < 0.0) | (beta > 1.0)):
            raise ValueError("selection audit beta must be finite and lie in [0, 1]")
        if not np.all(np.isfinite(redshift)):
            raise ValueError("selection audit redshift must be finite")
    elif contract.kind == "prior_evaluation":
        required = {"x", "log_p", "log_beta"}
        missing = required - set(arrays)
        if missing:
            raise ValueError(f"prior evaluation shard is missing: {sorted(missing)}")
        x = np.asarray(arrays["x"])
        log_p = np.asarray(arrays["log_p"])
        log_beta = np.asarray(arrays["log_beta"])
        if x.ndim != 2 or log_p.shape != x.shape[:1] or log_beta.shape != x.shape[:1]:
            raise ValueError("prior evaluation arrays have incompatible shapes")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(log_p)):
            raise ValueError("prior evaluation contains non-finite prior draws")
        if np.any(np.isnan(log_beta)) or np.any(log_beta > 0.0):
            raise ValueError("prior evaluation log_beta must be <= 0 and not NaN")


def _tree_l2_norm(tree) -> jnp.ndarray:
    leaves = [
        value
        for value in jax.tree_util.tree_leaves(tree)
        if value is not None and eqx.is_inexact_array(value)
    ]
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.square(value)) for value in leaves))


def _tree_all_finite(tree) -> jnp.ndarray:
    leaves = [
        value
        for value in jax.tree_util.tree_leaves(tree)
        if value is not None and eqx.is_inexact_array(value)
    ]
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves]))


def _pmean_tree(tree, axis_name: str):
    return jax.tree_util.tree_map(
        lambda value: jax.lax.pmean(value, axis_name) if value is not None else None,
        tree,
    )


def _select_tree(proposed, current, condition):
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(condition, new, old) if eqx.is_array(new) else new,
        proposed,
        current,
    )


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
