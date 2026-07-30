from __future__ import annotations

from pathlib import Path

from euclid_dsps.amortized.config import amortized_config
from euclid_dsps.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/popcosmos_native15d_rws.yaml"

NATIVE_PARAMETERS = (
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
    "sfh_dlog_sfr_01",
    "sfh_dlog_sfr_02",
    "sfh_dlog_sfr_03",
    "sfh_dlog_sfr_04",
    "sfh_dlog_sfr_05",
    "sfh_dlog_sfr_06",
    "sfh_dlog_sfr_07",
    "sfh_dlog_sfr_08",
    "sfh_dlog_sfr_09",
    "sfh_dlog_sfr_10",
)


def test_popcosmos_native_config_changes_observations_not_latent_physics() -> None:
    config = load_config(CONFIG)
    amortized = amortized_config(config)
    a24 = load_config(ROOT / "configs/experiments/popcosmos_a24_rws_joint.yaml")

    assert tuple(config["fit"]["free_parameters"]) == NATIVE_PARAMETERS
    assert config["model"]["sfh_model"] == "spline15d"
    assert config["model"]["nebular_model"] == "fixed_ssp"
    assert config["model"]["agn_model"] == "none"
    assert config["truth"]["parameter_columns"] == {}
    assert config["truth"]["redshift_column"] == "redshift_true"
    assert [band["name"] for band in config["bands"]] == [
        band["name"] for band in a24["bands"]
    ]

    assert amortized["latent"]["schema"] == "feniks_spline15d"
    assert amortized["latent"]["normalization"] == "spline15d_checkpoint"
    assert amortized["encoder"]["input_dim"] == 52
    assert amortized["encoder"]["latent_dim"] == 15
    assert amortized["likelihood"] == {
        "type": "student_t",
        "student_t_dof": 2.0,
        "error_floor_frac": 0.0,
        "error_jitter": 0.0,
    }
    assert amortized["objective"]["mode"] == "reweighted_wake_sleep"
    assert amortized["objective"]["wake"]["n_particles"] == 8
    assert amortized["objective"]["wake"]["start_encoder_epoch"] == 4
    assert amortized["objective"]["wake"]["every_encoder_epochs"] == 4


def test_native_map_wrapper_omits_amortized_checkpoint() -> None:
    script = (ROOT / "scripts/popcosmos_native15d_map_h100.slurm").read_text()
    assert "--prior-weight 0" in script
    assert "--start-mode latin_hypercube" in script
    assert "--checkpoint" not in script
    assert "map_normalized_residuals_by_band.csv" in script
