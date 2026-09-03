#!/usr/bin/env python3
"""Evaluate one population-flow candidate on frozen truth-free validation banks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.amortized.latent import latent_spec_from_config
from euclid_dsps.amortized.population_projection import (
    evaluate_log_beta,
    require_projection_runtime_commit,
    selection_runtime,
)
from euclid_dsps.amortized.population_projection_benchmark import (
    BASELINE_NAME,
    summarize_truth_free_metrics,
)
from euclid_dsps.amortized.population_vem import resolve_manifest_config, sha256_file
from euclid_dsps.amortized.train import load_checkpoint
from euclid_dsps.config import load_config

try:
    from scripts.evaluate_feniks_sc_drws_population_projection import (
        _comparison_rows,
        _load_parent_target,
        _load_q,
        _plot_physical_projection,
        _plot_redshift_projection,
        _sample_prior,
        _write_json,
        _x_to_theta,
    )
except ModuleNotFoundError:
    from evaluate_feniks_sc_drws_population_projection import (
        _comparison_rows,
        _load_parent_target,
        _load_q,
        _plot_physical_projection,
        _plot_redshift_projection,
        _sample_prior,
        _write_json,
        _x_to_theta,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_record(
    root: Path, manifest: dict[str, Any], index: int
) -> tuple[str, str, dict[str, Any], Path]:
    benchmark = manifest["architecture_benchmark"]
    candidates = benchmark["trained_candidates"]
    if index == len(candidates):
        baseline = benchmark["baseline"]
        return (
            BASELINE_NAME,
            baseline["label"],
            {
                "status": "COMPLETE",
                "candidate": BASELINE_NAME,
                "selected": baseline["selected"],
                "parent": baseline["parent"],
                "truth_used": False,
            },
            Path(baseline["config"]),
        )
    if not 0 <= index < len(candidates):
        raise ValueError("candidate evaluation index is outside the benchmark")
    candidate = candidates[index]
    receipt = _read_json(root / "candidates" / candidate["name"] / "FIT_COMPLETE.json")
    config_path = Path(candidate["config"])
    if sha256_file(config_path) != candidate["config_sha256"]:
        raise ValueError("candidate config SHA256 mismatch")
    return candidate["name"], candidate["label"], receipt, config_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = _read_json(root / "RUN_MANIFEST.json")
    repo = Path(__file__).resolve().parents[1]
    runtime_provenance = require_projection_runtime_commit(
        root, manifest, repo, stage="evaluation"
    )
    name, label, fit, config_path = _candidate_record(
        root, manifest, int(args.candidate_index)
    )
    if fit.get("status") != "COMPLETE" or fit.get("truth_used") is not False:
        raise ValueError("candidate fit is not complete and truth-free")
    candidate_root = root / "candidates" / name
    receipt_path = candidate_root / "TRUTH_FREE_EVALUATION.json"
    if receipt_path.is_file():
        print(receipt_path.read_text(encoding="utf-8"), flush=True)
        return

    config = load_config(config_path)
    base_config = load_config(resolve_manifest_config(manifest, "config", repo))
    latent_spec = latent_spec_from_config(config)
    if tuple(latent_spec.names) != tuple(latent_spec_from_config(base_config).names):
        raise ValueError("candidate changed the frozen latent specification")
    selected_checkpoint = Path(fit["selected"]["checkpoint"])
    parent_checkpoint = Path(fit["parent"]["checkpoint"])
    for record, checkpoint in (
        (fit["selected"], selected_checkpoint),
        (fit["parent"], parent_checkpoint),
    ):
        if sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError(f"candidate checkpoint SHA256 mismatch: {checkpoint}")
    selected_model = load_checkpoint(selected_checkpoint, config)
    parent_model = load_checkpoint(parent_checkpoint, config)

    q_x = _load_q(Path(manifest["q_banks"]["validation"]["manifest"]))
    parent_target_x, parent_target_weights = _load_parent_target(
        root / "banks" / "beta_validation" / "bank_manifest.json"
    )
    samples = int(manifest["request"]["prior_samples"])
    selected_x = _sample_prior(selected_model.prior, 282100, samples)
    parent_x = _sample_prior(parent_model.prior, 282200, samples)
    runtime = selection_runtime(base_config, Path(manifest["source"]["feature_stats"]))
    parent_log_beta = evaluate_log_beta(parent_model, parent_x, runtime, chunk_size=512)
    parent_beta = np.where(np.isfinite(parent_log_beta), np.exp(parent_log_beta), 0.0)
    if not np.isfinite(parent_beta.sum()) or parent_beta.sum() <= 0.0:
        raise ValueError("candidate parent flow has no finite selected mass")
    parent_selected_weights = parent_beta / parent_beta.sum()

    names = tuple(latent_spec.names)
    q_theta = _x_to_theta(q_x, latent_spec)
    parent_target_theta = _x_to_theta(parent_target_x, latent_spec)
    selected_theta = _x_to_theta(selected_x, latent_spec)
    parent_theta = _x_to_theta(parent_x, latent_spec)
    rows: list[dict[str, Any]] = []
    rows.extend(
        _comparison_rows(
            comparison="selected_flow_vs_q_aggregate",
            role="truth_free_validation_distribution",
            source=selected_theta,
            target=q_theta,
            names=names,
        )
    )
    rows.extend(
        _comparison_rows(
            comparison="parent_flow_vs_inverse_beta_q",
            role="truth_free_validation_distribution",
            source=parent_theta,
            target=parent_target_theta,
            names=names,
            target_weights=parent_target_weights,
        )
    )
    rows.extend(
        _comparison_rows(
            comparison="selected_parent_flow_vs_q_aggregate",
            role="truth_free_validation_distribution",
            source=parent_theta,
            target=q_theta,
            names=names,
            source_weights=parent_selected_weights,
        )
    )
    metrics = pd.DataFrame(rows)
    summary = summarize_truth_free_metrics(metrics, parameter_names=names)
    summary["fit_validation_weighted_nll_mean"] = float(
        0.5
        * (
            float(fit["selected"]["best_validation_weighted_nll"])
            + float(fit["parent"]["best_validation_weighted_nll"])
        )
    )

    attempt = (
        candidate_root
        / f".validation-attempt-{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    if attempt.exists():
        shutil.rmtree(attempt)
    plots = attempt / "plots"
    plots.mkdir(parents=True)
    metrics.to_csv(attempt / "distribution_metrics.csv", index=False)
    _plot_redshift_projection(
        plots / "redshift_distribution_projection.png",
        q_theta,
        selected_theta,
        parent_target_theta,
        parent_target_weights,
        parent_theta,
        parent_selected_weights,
    )
    _plot_physical_projection(
        plots / "physical_distribution_projection.png",
        names,
        q_theta,
        selected_theta,
        parent_target_theta,
        parent_target_weights,
        parent_theta,
        parent_selected_weights,
    )
    final = candidate_root / "truth_free_validation"
    if final.exists():
        raise FileExistsError(final)
    os.replace(attempt, final)
    receipt = {
        "status": "COMPLETE",
        "candidate": name,
        "label": label,
        **summary,
        "fit": {
            "selected": fit["selected"],
            "parent": fit["parent"],
        },
        "parent_selection": {
            "alpha_monte_carlo": float(np.mean(parent_beta)),
            "selected_weight_ess_fraction": float(
                np.square(parent_beta.sum())
                / np.square(parent_beta).sum()
                / len(parent_beta)
            ),
        },
        "prior_samples": samples,
        "truth_used": False,
        "point_estimates_used": False,
        "new_posterior_inference": False,
        "runtime_provenance": runtime_provenance,
        "artifacts": {
            "metrics": str((final / "distribution_metrics.csv").resolve()),
            "redshift_plot": str(
                (final / "plots" / "redshift_distribution_projection.png").resolve()
            ),
            "physical_plot": str(
                (final / "plots" / "physical_distribution_projection.png").resolve()
            ),
        },
    }
    receipt["artifact_sha256"] = {
        key: sha256_file(path) for key, path in receipt["artifacts"].items()
    }
    _write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
