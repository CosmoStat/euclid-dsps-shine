from __future__ import annotations

import pandas as pd

from euclid_dsps.amortized.prior_overlap import write_diffsky_prior_overlap_report


def test_population_overlap_includes_log_sfr_without_raw_sfh_ratio(tmp_path) -> None:
    dataset = tmp_path / "dataset.parquet"
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "report"
    run_dir.mkdir()

    pd.DataFrame(
        {
            "object_id": [1, 2],
            "redshift_true": [0.5, 1.0],
            "logsm_true": [9.5, 10.5],
            "logsfr_true": [0.0, 1.0],
            "diffstar_lgmcrit": [11.0, 11.2],
        }
    ).to_parquet(dataset, index=False)
    pd.DataFrame(
        {
            "object_id": [1, 1, 2, 2],
            "z_obs": [0.45, 0.55, 0.95, 1.05],
            "log10_stellar_mass": [9.4, 9.6, 10.4, 10.6],
            "log10_sfr_at_obs": [-0.1, 0.1, 0.9, 1.1],
            "dlog10_sfr_1": [2.0, 2.0, 2.0, 2.0],
        }
    ).to_parquet(run_dir / "posterior_samples.parquet", index=False)
    pd.DataFrame(
        {
            "object_id": [1, 2],
            "z_obs_median": [0.5, 1.0],
            "log10_stellar_mass_median": [9.5, 10.5],
            "log10_sfr_at_obs_median": [0.0, 1.0],
        }
    ).to_parquet(run_dir / "posterior_summary.parquet", index=False)
    pd.DataFrame(
        {
            "z_obs": [0.4, 0.6, 0.9, 1.1],
            "log10_stellar_mass": [9.3, 9.7, 10.3, 10.7],
            "log10_sfr_at_obs": [-0.2, 0.2, 0.8, 1.2],
            "diffstar_lgmcrit": [10.9, 11.1, 11.1, 11.3],
        }
    ).to_parquet(run_dir / "learned_or_loaded_prior_samples.parquet", index=False)

    write_diffsky_prior_overlap_report(
        dataset_path=dataset,
        run_dir=run_dir,
        out_dir=out_dir,
        config={},
    )

    metrics = pd.read_csv(out_dir / "prior_overlap_metrics.csv")
    assert "log_sfr_at_obs" in set(metrics["label"])
    assert "diffstar_lgmcrit" in set(metrics["label"])
    assert "dlog10_sfr_1" not in set(metrics["parameter"])
    assert (out_dir / "population_realism_report.md").exists()
