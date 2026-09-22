import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_stratified_indices_are_unique_and_include_both_selection_states():
    from scripts.feniks_failure_modes import _stratified_indices

    rows = np.arange(200)
    truth = pd.DataFrame(
        {"parent_row_index": rows, "z_obs": np.linspace(0.01, 5.49, len(rows))}
    )
    selection = pd.DataFrame({"parent_row_index": rows, "selected": rows % 3 != 0})
    chosen = _stratified_indices(truth, selection, 80, 17)
    assert len(chosen) == 80
    assert len(np.unique(chosen)) == 80
    states = selection.set_index("parent_row_index").loc[chosen, "selected"]
    assert set(states) == {False, True}


def test_weighted_quantile_tracks_population_weights():
    from scripts.feniks_failure_modes import _weighted_quantile

    values = np.array([0.0, 1.0, 10.0])
    uniform = _weighted_quantile(values, [0.5], np.ones(3))[0]
    weighted = _weighted_quantile(values, [0.5], [1.0, 1.0, 100.0])[0]
    assert uniform == 1.0
    assert weighted > 9.0


def test_posterior_metrics_use_named_physical_group():
    from scripts.feniks_failure_modes import _posterior_metrics

    names = (
        "sfh_dlog_sfr_01",
        "z_obs",
        "log10_stellar_mass",
        "log10_stellar_metallicity",
        "dust_av",
        "dust_delta",
    )
    truth = np.zeros((2, len(names)))
    draws = np.zeros((2, 8, len(names)))
    metadata = {
        "z_obs": np.zeros(2),
        "r_mag": np.full(2, 28.0),
        "log10_snr": np.ones(2),
        "selection_margin_sigma": np.ones(2),
    }
    frame = _posterior_metrics(draws, truth, metadata, names, "simulation", "test")
    groups = frame.groupby("parameter").group.first().to_dict()
    assert groups["sfh_dlog_sfr_01"] == "sfh"
    assert groups["z_obs"] == "physical"
    assert sum(value == "physical" for value in groups.values()) == 5


def test_report_decision_uses_continued_posterior_not_original(tmp_path):
    from scripts.feniks_failure_modes import TASKS, report, write

    settings = {
        "gates": {
            "decoder_p95_abs_residual_sigma": 0.25,
            "error_p95_relative_residual": 0.02,
            "noise_mean_abs": 0.08,
            "noise_std_abs_from_one": 0.08,
            "nuisance_population_selection_abs": 0.03,
            "nuisance_zbin_selection_abs": 0.08,
            "simulation_coverage68_abs": 0.03,
            "simulation_coverage95_abs": 0.02,
            "catalogue_coverage68_abs": 0.03,
            "catalogue_coverage95_abs": 0.02,
            "alpha_abs": 0.02,
            "known_mixture_max_w1_over_iqr": 0.10,
            "bootstrap_alpha_std": 0.02,
        }
    }
    config = tmp_path / "audit.yaml"
    config.write_text("gates: {}\n")
    inputs = {"config": {"path": str(config), "sha256": ""}}
    from scripts.feniks_avi_experiments import sha

    inputs["config"]["sha256"] = sha(config)
    write(tmp_path / "MANIFEST.json", {"settings": settings, "inputs": inputs})
    for task in TASKS:
        (tmp_path / task).mkdir()
    write(
        tmp_path / "decoder_observation/FINAL.json",
        {
            "status": "DECODER_OBSERVATION_AUDIT_COMPLETE",
            "selection_identity_mismatches": 0,
        },
    )
    write(
        tmp_path / "nuisance_sensitivity/FINAL.json",
        {
            "status": "NUISANCE_SENSITIVITY_COMPLETE",
            "population_selection_probability_abs_shift": 0.01,
            "max_zbin_selection_probability_abs_shift": 0.02,
        },
    )
    for arm in ("original", "continued"):
        write(
            tmp_path / f"posterior_{arm}/FINAL.json",
            {"status": "POSTERIOR_CONDITIONAL_AUDIT_COMPLETE"},
        )
    write(
        tmp_path / "population_identifiability/FINAL.json",
        {
            "status": "POPULATION_IDENTIFIABILITY_COMPLETE",
            "weighted_alpha_abs_error": 0.01,
            "known_mixture_max_physical_w1_over_iqr": 0.05,
        },
    )
    pd.DataFrame(
        {
            "band": ["lsst_r"],
            "decoder_p95_abs_sigma": [0.1],
            "error_true_p95_abs_relative": [0.01],
            "noise_mean": [0.0],
            "noise_std": [1.0],
        }
    ).to_csv(tmp_path / "decoder_observation/band_summary.csv", index=False)
    pd.DataFrame({"band": ["lsst_r"], "p90_abs_flux_shift_sigma": [0.1]}).to_csv(
        tmp_path / "nuisance_sensitivity/band_sensitivity.csv", index=False
    )
    for arm, coverage in (("original", 0.2), ("continued", 0.68)):
        summary = []
        conditional = []
        for cohort in ("simulation", "catalogue"):
            for parameter in (
                "z_obs",
                "log10_stellar_mass",
                "log10_stellar_metallicity",
                "dust_av",
                "dust_delta",
            ):
                summary.append(
                    {
                        "arm": arm,
                        "cohort": cohort,
                        "group": "physical",
                        "parameter": parameter,
                        "coverage68": coverage,
                        "coverage95": 0.95 if arm == "continued" else 0.3,
                    }
                )
            conditional.append(
                {
                    "arm": arm,
                    "cohort": cohort,
                    "parameter": "z_obs",
                    "conditioning_variable": "z_obs",
                    "bin": "(0, 1]",
                    "coverage68": coverage,
                }
            )
        pd.DataFrame(summary).to_csv(
            tmp_path / f"posterior_{arm}/marginal_summary.csv", index=False
        )
        pd.DataFrame(conditional).to_csv(
            tmp_path / f"posterior_{arm}/conditional_calibration.csv", index=False
        )
    pd.DataFrame(
        {
            "replicate": np.arange(6),
            "parameter": ["z_obs"] * 6,
            "w1_over_truth_iqr": np.full(6, 0.2),
            "alpha": np.linspace(0.90, 0.91, 6),
        }
    ).to_csv(tmp_path / "population_identifiability/bootstrap_metrics.csv", index=False)
    report(tmp_path)
    decision = json.loads((tmp_path / "DECISION.json").read_text())
    assert not decision["checks"]["original_catalogue_68"]
    assert decision["checks"]["continued_catalogue_68"]
    assert decision["ready_for_production"]
    assert (tmp_path / "failure_modes_summary.png").is_file()
    nuisance_final = json.loads(
        (tmp_path / "nuisance_sensitivity/FINAL.json").read_text()
    )
    nuisance_final["population_selection_probability_abs_shift"] = 0.5
    nuisance_final["max_zbin_selection_probability_abs_shift"] = 0.5
    write(tmp_path / "nuisance_sensitivity/FINAL.json", nuisance_final)
    report(tmp_path)
    decision = json.loads((tmp_path / "DECISION.json").read_text())
    assert not decision["checks"]["nuisance_conditional"]
    assert not decision["ready_for_production"]
    assert "parent_nuisance_selection_contract" in decision["failure_modes"]
    assert "individual SFH" in decision["check_interpretation"]["nuisance_conditional"]


def test_failure_mode_slurm_chain_is_bounded_and_fail_closed():
    submit = Path("scripts/submit_feniks_failure_modes.sh").read_text()
    worker = Path("scripts/feniks_failure_modes.slurm").read_text()
    watcher = Path("scripts/watch_feniks_failure_modes.sh").read_text()
    assert '--array="0-4%$CONCURRENCY"' in submit
    assert '--dependency="afterok:$ARRAY_JOB"' in submit
    assert "5 * ARRAY_HOURS + REPORT_HOURS" in submit
    assert "#SBATCH --gres=gpu:1" in worker
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in worker
    assert "Population + classifier" in watcher
    assert "ready for production" in watcher
