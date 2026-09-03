from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from euclid_dsps.amortized.config import require_amortized_dependencies
from euclid_dsps.amortized.population_projection import (
    distribution_comparison,
    inverse_selection_weights,
    make_pmap_weighted_density_step,
    require_projection_runtime_commit,
    uniform_cdf_distance,
    weighted_cdf_distance,
    weighted_cdf_values,
)
from euclid_dsps.amortized.population_projection_benchmark import (
    CORE_PARAMETER_NAMES,
    TRAINED_CANDIDATES,
    config_for_candidate,
    select_truth_free_candidate,
    summarize_truth_free_metrics,
)
from euclid_dsps.amortized.population_vem import (
    ArrayShardContract,
    is_array_bank_shard_complete,
    merge_array_bank_shards,
    sha256_file,
    write_array_bank_shard,
)
from scripts.evaluate_redshift_pit_coverage import finite_rank_pit, uniform_ks
from scripts.prepare_feniks_sc_drws_population_projection import (
    _validate_source_bank,
)
from scripts.prepare_feniks_sc_drws_population_projection_continuation import (
    prepare_continuation,
)

eqx, optax = require_amortized_dependencies()

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64


class _LocationNormalPrior(eqx.Module):
    location: jax.Array
    latent_dim: int = eqx.field(static=True)
    activation: Callable = jax.nn.tanh

    def log_prob(self, value):
        return -0.5 * jnp.sum(jnp.square(value - self.location), axis=-1)


def test_inverse_selection_weights_are_joint_draw_weights() -> None:
    weights, diagnostics = inverse_selection_weights(np.log([0.5, 1.0]))
    assert np.allclose(weights, [4.0 / 3.0, 2.0 / 3.0])
    assert np.isclose(diagnostics["alpha_harmonic"], 2.0 / 3.0)
    assert np.isclose(diagnostics["ess"], 1.8)
    assert diagnostics["weight_contract"].startswith("joint-draw weights")
    with pytest.raises(ValueError, match="beta=0"):
        inverse_selection_weights(np.asarray([-np.inf, 0.0]))


def test_distribution_comparisons_use_complete_weighted_cdfs() -> None:
    reference = np.asarray([0.0, 1.0, 2.0, 3.0])
    query = np.asarray([-1.0, 0.0, 1.5, 4.0])
    assert np.allclose(weighted_cdf_values(reference, query), [0.0, 0.25, 0.5, 1.0])
    assert weighted_cdf_distance(reference, reference) == 0.0
    shifted = distribution_comparison(reference + 0.5, reference)
    assert shifted["cdf_supremum"] > 0.0
    assert shifted["wasserstein"] > 0.0
    assert "distribution_rank_uniform_ks" in shifted


def test_uniform_cdf_distance_clamps_only_floating_point_roundoff() -> None:
    values = np.asarray([0.0, 0.5, np.nextafter(1.0, 2.0)])
    assert np.isfinite(uniform_cdf_distance(values))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        uniform_cdf_distance(np.asarray([0.0, 1.0 + 1.0e-8]))


def test_uniform_cdf_distance_keeps_weights_aligned_with_sorted_ranks() -> None:
    ranks = np.asarray([0.1, 0.5, 0.9])
    weights = np.asarray([0.2, 0.7, 0.1])
    assert uniform_cdf_distance(ranks, weights) == pytest.approx(0.4)


def test_matching_redshift_aggregate_does_not_imply_posterior_calibration() -> None:
    truth = np.asarray([0.0, 1.0])
    posterior = np.asarray([[1.0] * 32, [0.0] * 32])
    assert weighted_cdf_distance(posterior.reshape(-1), np.repeat(truth, 32)) == 0.0
    pit = finite_rank_pit(posterior, truth)
    assert uniform_ks(pit) > 0.45


def test_truth_free_architecture_score_excludes_sfh_and_redshift_medians() -> None:
    names = CORE_PARAMETER_NAMES + tuple(f"sfh_{index}" for index in range(10))
    rows = []
    for comparison in (
        "selected_flow_vs_q_aggregate",
        "parent_flow_vs_inverse_beta_q",
        "selected_parent_flow_vs_q_aggregate",
    ):
        for index, name in enumerate(names):
            rows.append(
                {
                    "comparison": comparison,
                    "parameter": name,
                    "cdf_supremum": 0.04 if index < 5 else 0.95,
                }
            )
    summary = summarize_truth_free_metrics(pd.DataFrame(rows), parameter_names=names)

    assert summary["passes_all_truth_free_distribution_gates"]
    assert summary["sfh_used_for_architecture_selection"] is False
    assert summary["redshift_median_gate_used"] is False
    assert summary["primary_score"] == pytest.approx(0.8)


def test_truth_free_candidate_selection_is_lexicographic() -> None:
    template = {
        "status": "COMPLETE",
        "truth_used": False,
        "sfh_used_for_architecture_selection": False,
        "redshift_median_gate_used": False,
        "fit_validation_weighted_nll_mean": 30.0,
        "passes_nll_non_regression_gate": True,
    }
    records = [
        {
            **template,
            "candidate": "lower_nll_but_worse_core",
            "primary_score": 1.2,
            "secondary_mean_core_5d_cdf_supremum": 0.08,
            "fit_validation_weighted_nll_mean": 20.0,
        },
        {
            **template,
            "candidate": "core_winner",
            "primary_score": 1.0,
            "secondary_mean_core_5d_cdf_supremum": 0.09,
        },
    ]
    assert select_truth_free_candidate(records)["candidate"] == "core_winner"


def test_truth_free_candidate_selection_rejects_joint_nll_regression() -> None:
    template = {
        "status": "COMPLETE",
        "truth_used": False,
        "sfh_used_for_architecture_selection": False,
        "redshift_median_gate_used": False,
        "secondary_mean_core_5d_cdf_supremum": 0.1,
    }
    records = [
        {
            **template,
            "candidate": "marginal_but_bad_density",
            "primary_score": 0.8,
            "fit_validation_weighted_nll_mean": 70.0,
            "passes_nll_non_regression_gate": False,
        },
        {
            **template,
            "candidate": "admissible_density",
            "primary_score": 1.5,
            "fit_validation_weighted_nll_mean": 29.0,
            "passes_nll_non_regression_gate": True,
        },
    ]

    assert select_truth_free_candidate(records)["candidate"] == "admissible_density"


def test_candidate_config_changes_only_prior_architecture() -> None:
    base = {"amortized": {"prior": {"source": "joint_realnvp"}}, "model": {"x": 1}}
    candidate = {"prior": {"source": "structured_rq_spline", "core_dim": 5}}
    result = config_for_candidate(base, candidate)
    assert result["amortized"]["prior"]["source"] == "structured_rq_spline"
    assert result["model"] == base["model"]
    assert base["amortized"]["prior"]["source"] == "joint_realnvp"
    assert all(
        candidate["prior"]["permutation"] == "roll" for candidate in TRAINED_CANDIDATES
    )


def test_q_beta_bank_preserves_draw_axis_and_selection_contract(tmp_path: Path) -> None:
    contract = ArrayShardContract(
        kind="q_beta_fit",
        dataset_sha256=SHA,
        checkpoint_sha256="b" * 64,
        latent_transform_sha256="c" * 64,
        code_commit="deadbeef",
        truth_used=False,
        draws_per_object=2,
        feature_stats_sha256="d" * 64,
        row_indices_sha256="e" * 64,
        selection_event="A=1[m_r_observed<29.0]",
    )
    arrays = {
        "row_index": np.asarray([7, 9], dtype=np.int64),
        "draw_index": np.asarray([0, 4], dtype=np.int64),
        "x": np.ones((2, 2, 3), dtype=np.float32),
        "log_q": np.zeros((2, 2), dtype=np.float32),
        "log_beta": np.log(np.full((2, 2), 0.8)),
    }
    write_array_bank_shard(tmp_path / "beta", 0, arrays, contract)
    assert is_array_bank_shard_complete(
        tmp_path / "beta" / "shards" / "shard_00000", validate_arrays=True
    )
    invalid = {**arrays, "draw_index": np.asarray([0, 0])}
    with pytest.raises(ValueError, match="unique source draws"):
        write_array_bank_shard(tmp_path / "invalid", 0, invalid, contract)


def test_projection_accepts_complete_truth_free_source_q_bank(tmp_path: Path) -> None:
    root = tmp_path / "source"
    rows = np.asarray([3, 7], dtype=np.int64)
    cohort = root / "q_fit.npy"
    cohort.parent.mkdir(parents=True)
    np.save(cohort, rows, allow_pickle=False)
    contract = ArrayShardContract(
        kind="q_train",
        dataset_sha256=SHA,
        checkpoint_sha256="b" * 64,
        latent_transform_sha256="c" * 64,
        code_commit="source",
        truth_used=False,
        draws_per_object=3,
        feature_stats_sha256="d" * 64,
        row_indices_sha256=sha256_file(cohort),
    )
    write_array_bank_shard(
        root / "banks" / "q_fit",
        0,
        {
            "row_index": rows,
            "x": np.ones((2, 3, 4), dtype=np.float32),
            "log_q": np.zeros((2, 3), dtype=np.float32),
        },
        contract,
    )
    merge_array_bank_shards(root / "banks" / "q_fit", expected_shards=1)
    source_manifest = {
        "frozen_source": {"checkpoint_sha256": "b" * 64},
        "banks": {
            "q_fit": {
                "draws_per_object": 3,
                "shards": 1,
                "objects": 2,
                "cohort_path": str(cohort),
                "cohort_sha256": sha256_file(cohort),
            }
        },
    }
    record = _validate_source_bank(root, source_manifest, "q_fit", "q_train")
    assert record["objects"] == 2
    assert record["draws_per_object"] == 3


def test_weighted_density_step_moves_toward_weighted_joint_draws() -> None:
    devices = tuple(jax.local_devices())
    count = len(devices)
    prior = _LocationNormalPrior(jnp.zeros(2), latent_dim=2)
    optimizer = optax.sgd(0.1)
    state = optimizer.init(eqx.filter(prior, eqx.is_inexact_array))

    def replicate(tree):
        return jax.tree_util.tree_map(
            lambda value: (
                jnp.broadcast_to(value, (count, *value.shape))
                if eqx.is_array(value)
                else value
            ),
            tree,
        )

    step = make_pmap_weighted_density_step(optimizer=optimizer)
    x = jnp.ones((count, 4, 2))
    weight = jnp.asarray(np.tile([2.0, 1.0, 0.5, 0.5], (count, 1)))
    valid = jnp.ones((count, 4), dtype=jnp.bool_)
    updated, _state, metrics = step(
        replicate(prior), replicate(state), x, weight, valid
    )
    assert np.all(np.asarray(metrics.update_applied))
    assert np.allclose(np.asarray(updated.location), 0.1)
    assert np.allclose(np.asarray(metrics.raw_gradient_norm), np.sqrt(2.0))


def test_projection_runtime_accepts_narrow_code_recovery(tmp_path: Path) -> None:
    runtime_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest_commit = "0" * 40
    manifest = {"code_commit": manifest_commit}
    manifest_path = tmp_path / "RUN_MANIFEST.json"
    beta_path = tmp_path / "BETA_TARGET_COMPLETE.json"
    manifest_path.write_text(json.dumps(manifest))
    beta_path.write_text(json.dumps({"status": "PASS", "truth_used": False}))
    (tmp_path / "SUBMISSION.json").write_text(json.dumps({"fit_job": "1709290"}))
    recovery = {
        "status": "AUTHORIZED",
        "scope": "fit_and_evaluation_only",
        "projection_root": str(tmp_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "beta_receipt_sha256": sha256_file(beta_path),
        "manifest_code_commit": manifest_commit,
        "runtime_code_commit": runtime_commit,
        "failed_fit_job": "1709290",
        "beta_banks_reused": True,
        "truth_used": False,
    }
    recovery_path = tmp_path / "CODE_RECOVERY.json"
    recovery_path.write_text(json.dumps(recovery))

    provenance = require_projection_runtime_commit(
        tmp_path, manifest, ROOT, stage="fit"
    )
    assert provenance["mode"] == "authorized_recovery"
    assert provenance["runtime_code_commit"] == runtime_commit

    recovery["truth_used"] = True
    recovery_path.write_text(json.dumps(recovery))
    with pytest.raises(ValueError, match="truth_used"):
        require_projection_runtime_commit(tmp_path, manifest, ROOT, stage="fit")


def test_projection_runtime_accepts_evaluation_only_recovery(tmp_path: Path) -> None:
    runtime_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest_commit = "0" * 40
    manifest = {"code_commit": manifest_commit}
    manifest_path = tmp_path / "RUN_MANIFEST.json"
    beta_path = tmp_path / "BETA_TARGET_COMPLETE.json"
    fit_path = tmp_path / "PROJECTION_FIT_COMPLETE.json"
    manifest_path.write_text(json.dumps(manifest))
    beta_path.write_text(json.dumps({"status": "PASS", "truth_used": False}))
    fit_path.write_text(json.dumps({"status": "COMPLETE", "truth_used": False}))
    (tmp_path / "SUBMISSION.json").write_text(json.dumps({"fit_job": "1709290"}))
    code_recovery = {
        "status": "AUTHORIZED",
        "scope": "fit_and_evaluation_only",
        "projection_root": str(tmp_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "beta_receipt_sha256": sha256_file(beta_path),
        "manifest_code_commit": manifest_commit,
        "runtime_code_commit": "1" * 40,
        "failed_fit_job": "1709290",
        "beta_banks_reused": True,
        "truth_used": False,
    }
    code_recovery_path = tmp_path / "CODE_RECOVERY.json"
    code_recovery_path.write_text(json.dumps(code_recovery))
    (tmp_path / "RECOVERY_SUBMISSION.json").write_text(
        json.dumps({"recovery_evaluation_job": "1710543"})
    )
    evaluation_recovery = {
        "status": "AUTHORIZED",
        "scope": "evaluation_only",
        "projection_root": str(tmp_path.resolve()),
        "code_recovery_sha256": sha256_file(code_recovery_path),
        "fit_receipt_sha256": sha256_file(fit_path),
        "failed_evaluation_job": "1710543",
        "runtime_code_commit": runtime_commit,
        "fit_reused": True,
        "truth_used": False,
    }
    (tmp_path / "EVALUATION_CODE_RECOVERY.json").write_text(
        json.dumps(evaluation_recovery)
    )

    provenance = require_projection_runtime_commit(
        tmp_path, manifest, ROOT, stage="evaluation"
    )
    assert provenance["mode"] == "authorized_evaluation_recovery"
    assert provenance["runtime_code_commit"] == runtime_commit


def test_projection_continuation_reuses_truth_free_checkpoints_and_banks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in ("beta_fit", "beta_validation"):
        bank = source / "banks" / name
        bank.mkdir(parents=True)
        (bank / "bank_manifest.json").write_text(
            json.dumps({"status": "complete", "name": name})
        )
    records = {}
    for label, target in (
        ("selected", "selected_q_aggregate"),
        ("parent", "parent_inverse_beta_q"),
    ):
        checkpoint = source / f"{label}.eqx"
        sidecar = source / f"{label}.eqx.json"
        checkpoint.write_bytes(label.encode())
        sidecar.write_text(json.dumps({"label": label}))
        records[label] = {
            "target": target,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_sidecar": str(sidecar),
            "checkpoint_sidecar_sha256": sha256_file(sidecar),
            "best_validation_weighted_nll": 12.0,
            "passes_completed": 16,
        }
    (source / "RUN_MANIFEST.json").write_text(
        json.dumps({"status": "PREPARED", "resources": {"beta_tasks": 20}})
    )
    (source / "BETA_TARGET_COMPLETE.json").write_text(
        json.dumps({"status": "PASS", "truth_used": False})
    )
    (source / "PROJECTION_FIT_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "truth_used": False,
                "point_estimates_used": False,
                "checkpoint_selection": "held-out weighted density only",
                **records,
            }
        )
    )

    out = tmp_path / "continuation"
    manifest = prepare_continuation(
        source_root=source,
        out=out,
        repo=ROOT,
        passes=48,
        patience=8,
        peak_learning_rate=1.0e-5,
        final_learning_rate=5.0e-7,
    )

    assert manifest["continuation"]["truth_used"] is False
    assert manifest["continuation"]["optimizer_state_reused"] is False
    assert manifest["resources"]["beta_tasks"] == 0
    assert manifest["resources"]["new_posterior_inference"] is False
    assert (out / "banks" / "beta_fit").is_symlink()
    assert (out / "banks" / "beta_validation").is_symlink()
    assert (out / "SOURCE_PROJECTION_FIT_COMPLETE.json").is_file()


def test_projection_submission_reuses_banks_and_separates_pit() -> None:
    submit = (
        ROOT / "scripts/submit_feniks_sc_drws_population_projection.sh"
    ).read_text()
    evaluate = (
        ROOT / "scripts/evaluate_feniks_sc_drws_population_projection.py"
    ).read_text()
    monitor = (
        ROOT / "scripts/monitor_feniks_sc_drws_population_projection.sh"
    ).read_text()
    recovery = (
        ROOT / "scripts/recover_feniks_sc_drws_population_projection.sh"
    ).read_text()
    evaluation_recovery = (
        ROOT / "scripts/recover_feniks_sc_drws_population_projection_evaluation.sh"
    ).read_text()
    continuation = (
        ROOT / "scripts/submit_feniks_sc_drws_population_projection_continuation.sh"
    ).read_text()
    assert "--array=0-19%20" in submit
    assert (
        "--gres=gpu:4"
        in (
            ROOT / "scripts/feniks_sc_drws_population_projection_fit_h100.slurm"
        ).read_text()
    )
    assert (
        "new_posterior_inference"
        in (
            ROOT / "scripts/prepare_feniks_sc_drws_population_projection.py"
        ).read_text()
    )
    assert "evaluate_redshift" in evaluate
    assert '"redshift_median_gate_used": False' in evaluate
    assert "distribution ranks are not posterior PIT" in monitor
    assert "fit_and_evaluation_only" in recovery
    assert "beta_banks_reused" in recovery
    assert "evaluation_only" in evaluation_recovery
    assert "fit_reused" in evaluation_recovery
    assert "new_posterior_inference" in continuation
    assert "beta_banks_reused" in continuation
    assert 'FIT_PASSES="${FIT_PASSES:-48}"' in continuation
    assert "population_projection_beta_h100.slurm" not in continuation
    assert 'export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"' in continuation
    assert "active-checkout import required" in continuation
    assert (
        "retain_initial_if_best=continuation is not None"
        in (ROOT / "scripts/train_feniks_sc_drws_population_projection.py").read_text()
    )


def test_projection_architecture_benchmark_is_parallel_and_truth_separated() -> None:
    submit = (
        ROOT / "scripts/submit_feniks_sc_drws_population_projection_benchmark.sh"
    ).read_text()
    prepare = (
        ROOT / "scripts/prepare_feniks_sc_drws_population_projection_benchmark.py"
    ).read_text()
    monitor = (
        ROOT / "scripts/monitor_feniks_sc_drws_population_projection_benchmark.sh"
    ).read_text()

    assert "--array=0-2%3" in submit
    assert "--array=0-3%4" in submit
    assert 'dependency="afterok:$GATE_JOB"' in submit
    assert '"new_posterior_inference": False' in submit
    assert '"truth_used_before_winner_freeze": False' in submit
    assert "closure_runs_after_winner_freeze" in prepare
    assert "redshift_median_gate_used" in prepare
    assert "maximum_validation_weighted_nll_regression" in prepare
    assert "q PIT KS" in monitor
