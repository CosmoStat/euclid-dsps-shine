from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from euclid_dsps.amortized.mira import FENIKS_SPLINE15D_PARAMETERS
from scripts.feniks_avi_em_factorial import (
    VARIANTS,
    _factorial_effects,
    _summary,
    prepare,
)
from scripts.feniks_avi_experiments import read, sha, write


def _fixture(tmp_path: Path, monkeypatch, *, cycles: int = 4) -> Path:
    root = tmp_path / "em"
    root.mkdir()
    for name in ("source_config.yaml", "train.npy", "validation.npy"):
        path = root / name
        if path.suffix == ".npy":
            np.save(path, np.array([1, 4]) if name == "validation.npy" else [0, 2])
        else:
            path.write_text("fixture: true\n")

    rng = np.random.default_rng(42)
    catalog = tmp_path / "selected.parquet"
    pd.DataFrame(
        rng.normal(size=(6, 15)), columns=FENIKS_SPLINE15D_PARAMETERS
    ).to_parquet(catalog, index=False)
    truth_paths = []
    for name, count in (("true_selected", 5), ("true_parent", 7)):
        frame = pd.DataFrame(
            rng.normal(size=(count, 15)), columns=FENIKS_SPLINE15D_PARAMETERS
        )
        frame.insert(0, "population_weight", np.arange(1, count + 1))
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        truth_paths.append(path)

    q0, p0, q4, p4 = (
        tmp_path / name for name in ("q0.eqx", "p0.eqx", "q4.eqx", "p4.eqx")
    )
    for path in (q0, p0, q4, p4):
        path.write_bytes(path.name.encode())
    cycle = root / "cycles/cycle_04"
    cycle.mkdir(parents=True)
    write(
        cycle / "FINAL.json",
        {
            "status": "EM_CYCLE_COMPLETE",
            "encoder": str(q4),
            "prior": str(p4),
            "encoder_sha256": sha(q4),
            "prior_sha256": sha(p4),
        },
    )
    manifest = {
        "cycles": cycles,
        "initial_encoder": str(q0),
        "initial_prior": str(p0),
        "validation_catalog": str(catalog),
        "true_selected": str(truth_paths[0]),
        "true_parent": str(truth_paths[1]),
        "source_training": "source-training",
        "source": {"checkpoint": "checkpoint.eqx", "feature_stats": "features.json"},
        "hashes": {},
    }
    write(root / "MANIFEST.json", manifest)
    monkeypatch.setattr("scripts.feniks_avi_em_factorial.POPULATION_SAMPLES", 32)
    monkeypatch.setattr(
        "scripts.feniks_avi_em_factorial._components",
        lambda _root, cycle_number: (q0, p0) if cycle_number == 0 else (q4, p4),
    )
    return root


def test_prepare_freezes_complete_q0_q4_by_p0_p4_factorial(tmp_path, monkeypatch):
    em = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "factorial"
    prepare(em, output, particles=4096)

    manifest = read(output / "MANIFEST.json")
    assert tuple(item["name"] for item in manifest["variants"]) == VARIANTS
    mapping = {
        item["name"]: (Path(item["encoder"]).name, Path(item["prior"]).name)
        for item in manifest["variants"]
    }
    assert mapping == {
        "Q0_P0": ("q0.eqx", "p0.eqx"),
        "Q4_P0": ("q4.eqx", "p0.eqx"),
        "Q0_P4": ("q0.eqx", "p4.eqx"),
        "Q4_P4": ("q4.eqx", "p4.eqx"),
    }
    assert manifest["particles"] == 4096
    assert manifest["replicas"] == 2
    assert manifest["selection_in_object_weights"] is False
    assert manifest["truth_used_for_sampling_or_weighting"] is False
    assert len(pd.read_parquet(output / "selected_population_truth.parquet")) == 32
    assert len(pd.read_parquet(output / "parent_population_truth.parquet")) == 32
    inference_truth = pd.read_parquet(output / "inference_truth.parquet")
    assert inference_truth.row_index.tolist() == [1, 4]


def test_prepare_rejects_non_four_cycle_history(tmp_path, monkeypatch):
    em = _fixture(tmp_path, monkeypatch, cycles=3)
    with pytest.raises(ValueError, match="exactly four completed EM cycles"):
        prepare(em, tmp_path / "factorial", particles=4096)


def test_factorial_effects_include_conditional_deltas_and_interaction():
    cells = pd.DataFrame(
        {
            "variant": VARIANTS,
            "metric": [1.0, 3.0, 4.0, 10.0],
        }
    )
    effects = _factorial_effects(cells).set_index("effect")
    assert effects.loc["Q effect at P0", "delta"] == 2.0
    assert effects.loc["P effect at Q0", "delta"] == 3.0
    assert effects.loc["QxP interaction", "delta"] == 4.0


def test_factorial_launch_uses_four_gpu_inference_and_afterok_report():
    slurm = Path("scripts/feniks_avi_em_factorial.slurm").read_text()
    submit = Path("scripts/submit_feniks_avi_em_factorial.sh").read_text()
    assert "#SBATCH --gres=gpu:4" in slurm
    assert "--mem" not in slurm
    assert '--array="0-3%${CONCURRENCY}"' in submit
    assert '--dependency="afterok:$INFERENCE_JOB"' in submit


def test_factorial_summary_writes_cells_effects_and_plots(tmp_path):
    output = tmp_path / "factorial"
    (output / "report").mkdir(parents=True)
    rng = np.random.default_rng(91)
    truth = pd.DataFrame(rng.normal(size=(3, 15)), columns=FENIKS_SPLINE15D_PARAMETERS)
    truth.insert(0, "row_index", [4, 7, 9])
    truth.insert(0, "object_id", truth.row_index)
    truth.to_parquet(output / "inference_truth.parquet", index=False)
    for population in ("selected", "parent"):
        frame = pd.DataFrame(
            rng.normal(size=(30, 15)), columns=FENIKS_SPLINE15D_PARAMETERS
        )
        frame.to_parquet(output / f"{population}_population_truth.parquet", index=False)

    mira_rows = []
    for variant_index, variant in enumerate(VARIANTS):
        arm = output / "arms" / variant
        arm.mkdir(parents=True)
        metric_rows = []
        for replica in range(2):
            draws = np.repeat(
                truth[list(FENIKS_SPLINE15D_PARAMETERS)].to_numpy()[:, None, :],
                8,
                axis=1,
            ) + rng.normal(scale=0.2, size=(3, 8, 15))
            samples = pd.DataFrame(
                draws.reshape(-1, 15), columns=FENIKS_SPLINE15D_PARAMETERS
            )
            samples.insert(0, "sample_id", np.tile(np.arange(8), 3))
            samples.insert(0, "row_index", np.repeat([4, 7, 9], 8))
            samples.to_parquet(arm / f"is_{replica}.parquet", index=False)
            bank_dir = arm / f"bank_{replica}"
            bank_dir.mkdir()
            bank = samples.copy()
            bank.insert(2, "weight", np.tile(np.full(8, 1 / 8), 3))
            bank.insert(3, "log_beta", np.log(rng.uniform(0.6, 1.0, len(bank))))
            bank.to_parquet(bank_dir / "part_00000.parquet", index=False)
            metric_rows.append(
                {
                    "replica": replica,
                    "ess_fraction": 0.1 + 0.01 * variant_index,
                    "max_weight": 0.2,
                    "raw_predictive_rms": 0.6,
                    "is_predictive_rms": 0.5,
                    "log_evidence": -20.0,
                    "resampled_unique": 7,
                }
            )
            for kind in ("raw", "is"):
                for group in ("physical_5d", "sfh_contrasts_10d"):
                    mira_rows.append(
                        {
                            "replica": replica,
                            "model": f"{variant}_{kind}",
                            "group": group,
                            "score": 0.65 + 0.01 * variant_index,
                            "bootstrap_std": 0.01,
                        }
                    )
        pd.DataFrame(metric_rows).to_csv(arm / "metrics.csv", index=False)
    pd.DataFrame(mira_rows).to_csv(
        output / "report/posterior_mira_scores.csv", index=False
    )

    manifest = {
        "variants": [{"name": name} for name in VARIANTS],
        "replicas": 2,
    }
    write(output / "MANIFEST.json", manifest)
    _summary(output, manifest)
    assert set(pd.read_csv(output / "report/factorial_cells.csv").variant) == set(
        VARIANTS
    )
    assert set(pd.read_csv(output / "report/factorial_effects.csv").effect) == {
        "Q effect at P0",
        "Q effect at P4",
        "P effect at Q0",
        "P effect at Q4",
        "QxP interaction",
    }
    cohort = pd.read_csv(output / "report/factorial_cohort_baseline.csv")
    assert len(cohort) == 15
    assert cohort.cohort_truth_wasserstein_over_selected_truth_iqr.ge(0).all()
    assert (output / "report/factorial_population.png").stat().st_size > 10_000
