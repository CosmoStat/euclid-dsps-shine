from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.config import load_config
from scripts.run_feniks_exact_posterior_benchmark import (
    _adaptive_smc_benchmark_configs,
)
from scripts.summarize_feniks_smc_teacher_audit import summarize


def test_teacher_budget_is_extended_fallback_without_second_fallback(
    monkeypatch,
) -> None:
    @dataclass(frozen=True)
    class Config:
        n_particles: int
        max_stages: int

    proposal = object()
    monkeypatch.setattr(
        "scripts.run_feniks_exact_posterior_benchmark.adaptive_smc_configs",
        lambda _config: (Config(64, 8), Config(128, 12), proposal),
    )
    primary, fallback, actual_proposal = _adaptive_smc_benchmark_configs(
        {}, mode="teacher"
    )
    assert primary.n_particles == 128
    assert primary.max_stages == 48
    assert fallback is None
    assert actual_proposal is proposal

    monkeypatch.undo()
    production = load_config(
        "configs/experiments/"
        "feniks_selfsup_adaptive_smcwake_parentprior_selection_r25.yaml"
    )
    primary, fallback, _proposal = _adaptive_smc_benchmark_configs(
        production, mode="teacher"
    )
    assert primary.n_particles == 128
    assert primary.max_stages == 48
    assert primary.steps_after_resample == 4
    assert primary.final_steps_at_beta1 == 2
    assert fallback is None


def _write_galaxy(root: Path, item, *, distilled: bool, rng) -> None:
    directory = root / "galaxies" / (
        f"{int(item.order):02d}_{item.example_key}_row{int(item.row_index)}"
    )
    directory.mkdir(parents=True)
    (directory / "prepare_manifest.json").write_text(
        json.dumps({"latent_spec": {"names": ["a", "b"]}})
    )
    nuts = rng.normal(size=(4000, 2))
    q = nuts.copy() if distilled else nuts + np.asarray([1.0, -1.0])
    pd.DataFrame(q, columns=["x_a", "x_b"]).to_parquet(
        directory / "encoder_samples.parquet", index=False
    )
    (directory / "importance_diagnostics.json").write_text(
        json.dumps(
            {
                "raw_ess_fraction": 0.20 if distilled else 0.01,
                "raw_max_weight": 0.20 if distilled else 0.99,
                "pareto_k": 0.4 if distilled else 2.0,
            }
        )
    )
    if distilled:
        return
    nuts_dir = directory / "nuts"
    nuts_dir.mkdir()
    pd.DataFrame(nuts, columns=["x_a", "x_b"]).to_parquet(
        nuts_dir / "samples.parquet", index=False
    )
    (nuts_dir / "diagnostics.json").write_text(
        json.dumps({"max_rhat": 1.0, "min_bulk_ess": 1000, "min_tail_ess": 900})
    )
    teacher = pd.DataFrame(nuts, columns=["x_a", "x_b"])
    teacher["smc_weight"] = 1.0 / len(teacher)
    teacher.to_parquet(
        directory / "adaptive_smc_weighted_samples.parquet", index=False
    )
    (directory / "adaptive_smc_diagnostics.json").write_text(
        json.dumps(
            {
                "fallback_attempted": False,
                "eligible_after_fallback": True,
                "primary": {
                    "beta_final": 1.0,
                    "mutation_acceptance": 0.25,
                    "ancestor_ess_fraction": 0.1,
                    "epsilon_squared_jump": 1.0,
                    "mixing_failure": False,
                },
            }
        )
    )


def test_teacher_audit_separates_teacher_and_q_gates(tmp_path: Path) -> None:
    bootstrap = tmp_path / "bootstrap"
    distilled = tmp_path / "distilled"
    cohort = pd.DataFrame(
        {
            "order": np.arange(8),
            "example_key": [f"g{index}" for index in range(8)],
            "row_index": np.arange(100, 108),
            "object_id": [f"id{index}" for index in range(8)],
        }
    )
    for root in (bootstrap, distilled):
        root.mkdir()
        cohort.to_parquet(root / "cohort.parquet", index=False)
    rng = np.random.default_rng(12)
    for item in cohort.itertuples(index=False):
        state = rng.bit_generator.state
        _write_galaxy(bootstrap, item, distilled=False, rng=rng)
        rng.bit_generator.state = state
        _write_galaxy(distilled, item, distilled=True, rng=rng)

    receipt = summarize(
        bootstrap_root=bootstrap,
        distilled_root=distilled,
        out=tmp_path / "summary",
    )

    assert receipt["status"] == "PASS"
    assert receipt["teacher_ready"]
    assert receipt["q_ready"]
    assert receipt["checks"]["nuts_converged"]
    assert receipt["checks"]["teacher_mean_agreement"]
    assert receipt["q_only_importance"]["distilled"][
        "median_ess_fraction"
    ] == 0.20
