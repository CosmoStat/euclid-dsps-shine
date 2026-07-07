from __future__ import annotations

from pathlib import Path

from euclid_dsps.config import load_config

DIFFSKY_SCIENCE_CONFIGS = (
    "configs/diffsky_dataset_hltds_04_14.yaml",
    "configs/diffsky_dataset_hltds_03_31_zmax335_m5depth.yaml",
    "configs/diffsky_synthetic_feniks_260617_50k.yaml",
    "configs/diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml",
    "configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml",
)


def test_diffsky_science_configs_default_to_student_t_nu2() -> None:
    for config_path in DIFFSKY_SCIENCE_CONFIGS:
        config = load_config(Path(config_path))
        assert config["fit"]["photometric_likelihood"] == "student_t", config_path
        assert float(config["fit"]["student_t_dof"]) == 2.0, config_path
        if config.get("amortized", {}).get("enabled"):
            likelihood = config["amortized"]["likelihood"]
            assert likelihood["type"] == "student_t", config_path
            assert float(likelihood["student_t_dof"]) == 2.0, config_path


def test_supervised_prior_configs_do_not_enable_alpha_sed() -> None:
    for config_path in (
        "configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml",
    ):
        config = load_config(Path(config_path))
        alpha = config["calibration"]["global_sed_scale"]
        assert alpha["enabled"] is False
        assert alpha["mode"] == "disabled"
