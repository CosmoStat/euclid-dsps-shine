from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_feniks_exact_posterior_benchmark import _write_run_comparison
from scripts.select_feniks_mclmc_pilot import CONFIGS

ROOT = Path(__file__).resolve().parents[1]


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
    finalize = (ROOT / "scripts" / "feniks_exact_finalize_h100.slurm").read_text()
    assert "MPLBACKEND=Agg" in prepare
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
