import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.run_feniks_exact_posterior_benchmark import (
    _all_finite_at_most,
    _sample_chunks,
    _selected_samplers,
    _truth_theta,
    _write_run_comparison,
)
from scripts.select_feniks_mclmc_pilot import CONFIGS

ROOT = Path(__file__).resolve().parents[1]


def test_runner_can_be_executed_as_a_direct_script() -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_feniks_exact_posterior_benchmark.py"),
            "--help",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "prepare-cohort" in result.stdout


def test_real_data_runtime_does_not_require_truth() -> None:
    runtime = SimpleNamespace(
        arrays=SimpleNamespace(truth=None),
        latent_spec=SimpleNamespace(names=("z_obs", "log10_stellar_mass")),
    )
    assert _truth_theta(runtime) is None


def test_convergence_gate_rejects_missing_or_nonfinite_diagnostics() -> None:
    assert _all_finite_at_most(pd.Series([1.0, 1.01]), 1.01)
    assert not _all_finite_at_most(pd.Series([1.0, None]), 1.01)
    assert not _all_finite_at_most(pd.Series([1.0, np.inf]), 1.01)
    assert not _all_finite_at_most(pd.Series(dtype=float), 1.01)


def test_pilot_grid_contains_nuts_agreement_and_unadjusted_energy_controls() -> None:
    labels = [row[0] for row in CONFIGS]
    samplers = [row[1] for row in CONFIGS]
    thinnings = [row[3] for row in CONFIGS]

    assert labels == [
        "adj_official_t1",
        "adj_jaxfli_t1",
        "unadj_jaxfli_e1e3",
        "unadj_jaxfli_e5e4",
        "adj_jaxfli_t4",
        "adj_jaxfli_t8",
        "adj_jaxfli_t16",
    ]
    assert samplers.count("mclmc_unadjusted") == 2
    assert thinnings[-3:] == [4, 8, 16]
    selector = (ROOT / "scripts" / "select_feniks_mclmc_pilot.py").read_text()
    assert "nuts_standardized_mean_distance" in selector
    assert "validation_passes_rhat_1_10" in selector


def test_smoke_and_full_submission_topology_is_dependency_gated() -> None:
    smoke = (
        ROOT / "scripts" / "submit_feniks_exact_posterior_smoke.sh"
    ).read_text()
    full = (ROOT / "scripts" / "submit_feniks_exact_posterior_full.sh").read_text()

    assert "--array=0-1%2" in smoke
    assert "--array=0-7%8" in smoke
    assert '--dependency="afterok:$nuts:$mclmc"' in smoke
    assert "requested_upper_bound_h100_hours=8.33" in smoke

    assert "--array=0-6%7" in full
    assert "--array=0-27%8" in full
    assert '--dependency="afterok:$pilot_nuts:$pilot_grid"' in full
    assert '--dependency="afterok:$pilot_validate"' in full
    assert '--dependency="afterok:$full_nuts:$full_mclmc"' in full
    assert "requested_upper_bound_h100_hours=825.00" in full
    assert "SMOKE_ROOT" in full
    recovery = (
        ROOT / "scripts" / "submit_feniks_exact_posterior_smoke_recovery.sh"
    ).read_text()
    assert "PREP_DONE" in recovery
    assert "feniks_exact_prepare_h100.slurm" not in recovery
    assert '--dependency="afterok:$nuts:$mclmc"' in recovery


def test_adaptation_probe_and_two_galaxy_pilot_are_isolated() -> None:
    probe = (ROOT / "scripts/submit_feniks_mclmc_adaptation_probe.sh").read_text()
    pilot = (
        ROOT / "scripts/submit_feniks_exact_posterior_two_galaxy_pilot.sh"
    ).read_text()

    assert "--array=0 --time=01:00:00" in probe
    assert "MCLMC_TUNE=10" in probe
    assert "SAMPLE_CHUNKS=10" in probe
    assert "requested_upper_bound_h100_hours=1.00" in probe

    assert '--cohort-file "$SMOKE_ROOT/cohort.csv"' in pilot
    assert "--array=0-7%8" in pilot
    assert pilot.count("--array=0-7%8") == 2
    assert '--dependency="afterok:$nuts:$mclmc"' in pilot
    assert "requested_upper_bound_h100_hours=108.33" in pilot
    assert "submit_feniks_exact_posterior_full.sh" not in pilot
    assert "--array=0-27" not in pilot


def test_two_galaxy_nuts_submission_has_no_mclmc_dependency() -> None:
    submission = (
        ROOT / "scripts/submit_feniks_exact_posterior_two_galaxy_nuts.sh"
    ).read_text()

    assert '--cohort-file "$SMOKE_ROOT/cohort.csv"' in submission
    assert "--array=0-1%2" in submission
    assert "--array=0-7%8" in submission
    assert "SAMPLER=nuts" in submission
    assert "FINAL_SAMPLERS=nuts" in submission
    assert '--dependency="afterok:$nuts"' in submission
    assert "SAMPLER=mclmc" not in submission
    assert "NUTS_MAX_DOUBLINGS" in submission
    assert "requested_upper_bound_h100_hours=60.33" in submission


def test_two_galaxy_nuts_recovery_reuses_preparation_and_caps_tree_depth() -> None:
    recovery = (
        ROOT
        / "scripts/submit_feniks_exact_posterior_two_galaxy_nuts_recovery.sh"
    ).read_text()

    assert "PREP_DONE" in recovery
    assert "feniks_exact_prepare_h100.slurm" not in recovery
    assert 'NUTS_WARMUP="${NUTS_WARMUP:-50}"' in recovery
    assert 'NUTS_MAX_DOUBLINGS="${NUTS_MAX_DOUBLINGS:-4}"' in recovery
    assert 'SAMPLE_CHUNKS="${SAMPLE_CHUNKS:-100}"' in recovery
    assert "missing_tasks" in recovery
    assert '--array="${array_spec}%${concurrency}"' in recovery
    assert "FINAL_SAMPLERS=nuts" in recovery
    assert "requested_upper_bound_h100_hours_at_most=36.33" in recovery


def test_two_galaxy_nuts_probe_gates_the_parallel_recovery() -> None:
    probe = (
        ROOT / "scripts/submit_feniks_exact_posterior_two_galaxy_nuts_probe.sh"
    ).read_text()

    assert "--array=0 --time=04:00:00" in probe
    assert 'NUTS_WARMUP="${NUTS_WARMUP:-50}"' in probe
    assert 'NUTS_MAX_DOUBLINGS="${NUTS_MAX_DOUBLINGS:-4}"' in probe
    assert 'SAMPLE_CHUNKS="${SAMPLE_CHUNKS:-100}"' in probe
    assert "requested_upper_bound_h100_hours=4.00" in probe


def test_batched_nuts_probe_and_finish_preserve_chain_artifact_contract() -> None:
    wrapper = (
        ROOT / "scripts/feniks_exact_nuts_batched_h100.slurm"
    ).read_text()
    probe = (
        ROOT / "scripts/submit_feniks_exact_posterior_nuts_batched_probe.sh"
    ).read_text()
    finish = (
        ROOT / "scripts/submit_feniks_exact_posterior_nuts_batched_finish.sh"
    ).read_text()

    assert "--gres=gpu:1" in wrapper
    assert "sample-nuts-batched" in wrapper
    assert "JAX_ENABLE_X64=true" in wrapper
    assert "CHAIN_INDICES" in wrapper
    assert "--array=0 --time=04:00:00" in probe
    assert "CHAIN_INDICES=1:2:3" in probe
    assert "chain_00/DONE" in probe
    assert "summarize_feniks_nuts_batched_probe.py" not in probe
    assert "--array=1 --time=04:00:00" in finish
    assert '--dependency="afterok:$nuts"' in finish
    assert "FINAL_SAMPLERS=nuts" in finish
    assert "requested_upper_bound_h100_hours=8.33" in finish


def test_two_galaxy_big_nuts_is_probe_gated_resumable_and_provenanced() -> None:
    submission = (
        ROOT
        / "scripts/submit_feniks_exact_posterior_two_galaxy_nuts_big.sh"
    ).read_text()

    assert "batched_probe_summary.json" in submission
    assert 'summary.get("status") != "passed"' in submission
    assert "batched_divergences" in submission
    assert "throughput_speedup" in submission
    assert 'NUTS_WARMUP="${NUTS_WARMUP:-200}"' in submission
    assert 'NUTS_MAX_DOUBLINGS="${NUTS_MAX_DOUBLINGS:-4}"' in submission
    assert (
        'SAMPLE_CHUNKS="${SAMPLE_CHUNKS:-'
        "100:100:100:100:100:100:100:100:100:100}\""
    ) in submission
    assert 'NUTS_TIME="${NUTS_TIME:-20:00:00}"' in submission
    assert 'os.link(src, dst)' in submission
    assert '"row_index": int(row.row_index)' in submission
    assert '"object_id": str(row.object_id)' in submission
    assert '"draws_per_galaxy": 4 * sum(chunks)' in submission
    assert '"resume_granularity": "completed sample chunk"' in submission
    assert "sed_draws.npz" in submission
    assert "corner_full15.png" in submission
    assert "posterior_method_agreement.png" in submission
    assert '--array="${array_spec}%${concurrency}"' in submission
    assert "feniks_exact_nuts_batched_h100.slurm" in submission
    assert '--dependency="afterok:$nuts"' in submission
    assert "FINAL_SAMPLERS=nuts" in submission


def test_selected_samplers_accepts_nuts_only_and_rejects_invalid_contract() -> None:
    assert _selected_samplers(SimpleNamespace(samplers="nuts")) == ("nuts",)
    assert _selected_samplers(SimpleNamespace(samplers="nuts,mclmc")) == (
        "nuts",
        "mclmc",
    )
    for value in ("", "mclmc", "nuts,nuts", "nuts,hmc"):
        try:
            _selected_samplers(SimpleNamespace(samplers=value))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid sampler contract: {value!r}")


def test_sample_chunks_accepts_slurm_safe_colon_separator() -> None:
    args = SimpleNamespace(sample_chunks="100:300", mode="pilot")
    assert _sample_chunks(args) == (100, 300)


def test_adaptation_probe_summary_accepts_finite_adjusted_chain(
    tmp_path: Path,
) -> None:
    root = tmp_path / "probe"
    chain = (
        root
        / "galaxies"
        / "01_typical_row1358"
        / "mclmc_adaptation_probe_t10_compat"
        / "chain_00"
    )
    chunks = chain / "chunks"
    chunks.mkdir(parents=True)
    (chain / "DONE").touch()
    (chain / "chain_manifest.json").write_text(
        json.dumps(
            {
                "adaptation_mode": "blackjax_three_phase",
                "tune_steps": 10,
                "actual_tuning_integrator_steps": 24,
                "step_size": 0.01,
                "L": 0.04,
                "integration_steps_per_transition": 4,
                "stored_samples": 10,
                "warmup_elapsed_s": 12.0,
                "total_elapsed_s": 20.0,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "acceptance_rate": np.full(10, 0.8),
            "is_divergent": np.zeros(10, dtype=bool),
            "energy": np.linspace(1.0, 1.1, 10),
        }
    ).to_parquet(chunks / "part_000000_info.parquet", index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/summarize_feniks_mclmc_adaptation_probe.py"),
            "--root",
            str(root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["status"] == "passed"
    assert summary["mean_acceptance"] == 0.8


def test_exact_wrappers_avoid_unreliable_jobscratch_and_use_headless_plots() -> None:
    for name in (
        "feniks_exact_prepare_h100.slurm",
        "feniks_exact_chain_h100.slurm",
        "feniks_exact_finalize_h100.slurm",
    ):
        content = (ROOT / "scripts" / name).read_text()
        assert "JOBSCRATCH" not in content
        assert "module load arch/h100" in content
    prepare = (ROOT / "scripts" / "feniks_exact_prepare_h100.slurm").read_text()
    chain = (ROOT / "scripts" / "feniks_exact_chain_h100.slurm").read_text()
    finalize = (ROOT / "scripts" / "feniks_exact_finalize_h100.slurm").read_text()
    assert "MPLBACKEND=Agg" in prepare
    assert "JAX_ENABLE_X64=true" in chain
    assert "NUTS_MAX_DOUBLINGS" in chain
    assert "MPLBACKEND=Agg" in finalize


def test_full_benchmark_keeps_the_shared_rws_checkpoint_contract() -> None:
    for name in (
        "submit_feniks_exact_posterior_smoke.sh",
        "submit_feniks_exact_posterior_full.sh",
    ):
        content = (ROOT / "scripts" / name).read_text()
        assert "feniks_selfsup_paper_rws_k8_t2_seed2.yaml" in content
        assert "feniks_selfsup_paper_v1/rws_k8_t2_seed2" in content
        assert "train/checkpoints/best.eqx" in content
        assert "train/feature_stats.json" in content


def test_run_comparison_preserves_restyleable_tables(tmp_path: Path) -> None:
    cohort = pd.DataFrame(
        [{"order": 0, "example_key": "example", "row_index": 12}]
    )
    galaxy = tmp_path / "galaxies" / "00_example_row12"
    (galaxy / "nuts").mkdir(parents=True)
    (galaxy / "mclmc").mkdir()
    (galaxy / "prepare_manifest.json").write_text(
        '{"latent_spec": {"names": ["z_obs", "mass"]}}',
        encoding="utf-8",
    )
    pd.DataFrame([{"z_obs": 0.4, "mass": 10.0}]).to_parquet(
        galaxy / "truth.parquet", index=False
    )
    pd.DataFrame(
        [
            {"objective": 2.0, "z_obs": 0.5, "mass": 10.1},
            {"objective": 1.0, "z_obs": 0.45, "mass": 10.05},
        ]
    ).to_parquet(galaxy / "map_solutions.parquet", index=False)
    rng = np.random.default_rng(4)
    for relative in (
        "encoder_samples.parquet",
        "importance_resampled_samples.parquet",
        "nuts/samples.parquet",
        "mclmc/samples.parquet",
    ):
        pd.DataFrame(
            {
                "z_obs": rng.normal(0.4, 0.05, 40),
                "mass": rng.normal(10.0, 0.2, 40),
            }
        ).to_parquet(galaxy / relative, index=False)
    pd.DataFrame(
        {
            "band": ["g", "r"],
            "flux": [2.0, 3.0],
            "flux_err": [0.2, 0.3],
            "mask": [True, True],
        }
    ).to_parquet(galaxy / "observation.parquet", index=False)
    rows = []
    for method in ("Truth", "MAP", "Encoder", "Encoder + IS", "NUTS", "MCLMC"):
        rows.extend(
            {
                "method": method,
                "band": band,
                "flux_q16": flux - 0.1,
                "flux_q50": flux,
                "flux_q84": flux + 0.1,
            }
            for band, flux in (("g", 2.0), ("r", 3.0))
        )
    pd.DataFrame(rows).to_parquet(
        galaxy / "photometric_predictions.parquet", index=False
    )

    _write_run_comparison(tmp_path, cohort)

    assert (tmp_path / "posterior_agreement.parquet").is_file()
    assert (tmp_path / "posterior_method_agreement.png").is_file()
    assert (tmp_path / "photometric_fit_metrics.parquet").is_file()
    assert (tmp_path / "photometric_fit_comparison.pdf").is_file()


def test_run_comparison_supports_nuts_only(tmp_path: Path) -> None:
    cohort = pd.DataFrame(
        [{"order": 0, "example_key": "example", "row_index": 12}]
    )
    galaxy = tmp_path / "galaxies" / "00_example_row12"
    (galaxy / "nuts").mkdir(parents=True)
    (galaxy / "prepare_manifest.json").write_text(
        '{"latent_spec": {"names": ["z_obs", "mass"]}}',
        encoding="utf-8",
    )
    pd.DataFrame([{"z_obs": 0.4, "mass": 10.0}]).to_parquet(
        galaxy / "truth.parquet", index=False
    )
    pd.DataFrame(
        [{"objective": 1.0, "z_obs": 0.45, "mass": 10.05}]
    ).to_parquet(galaxy / "map_solutions.parquet", index=False)
    rng = np.random.default_rng(4)
    for relative in (
        "encoder_samples.parquet",
        "importance_resampled_samples.parquet",
        "nuts/samples.parquet",
    ):
        pd.DataFrame(
            {
                "z_obs": rng.normal(0.4, 0.05, 40),
                "mass": rng.normal(10.0, 0.2, 40),
            }
        ).to_parquet(galaxy / relative, index=False)
    pd.DataFrame(
        {
            "band": ["g", "r"],
            "flux": [2.0, 3.0],
            "flux_err": [0.2, 0.3],
            "mask": [True, True],
        }
    ).to_parquet(galaxy / "observation.parquet", index=False)
    rows = []
    for method in ("Truth", "MAP", "Encoder", "Encoder + IS", "NUTS"):
        rows.extend(
            {
                "method": method,
                "band": band,
                "flux_q16": flux - 0.1,
                "flux_q50": flux,
                "flux_q84": flux + 0.1,
            }
            for band, flux in (("g", 2.0), ("r", 3.0))
        )
    pd.DataFrame(rows).to_parquet(
        galaxy / "photometric_predictions.parquet", index=False
    )

    _write_run_comparison(tmp_path, cohort, samplers=("nuts",))

    agreement = pd.read_parquet(tmp_path / "posterior_agreement.parquet")
    assert set(agreement["method"]) == {"Encoder", "Encoder + IS", "NUTS", "MAP"}
    assert (tmp_path / "posterior_method_agreement.png").is_file()
