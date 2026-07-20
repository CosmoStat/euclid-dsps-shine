from pathlib import Path

import pytest

from euclid_dsps.amortized.config import amortized_config
from euclid_dsps.amortized.elbo import objective_uses_truth
from euclid_dsps.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
REFERENCE = "outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx"

CONFIGS = (
    "feniks_mode_common15d_vem4_elbo_k1.yaml",
    "feniks_mode_frozen_ref_elbo_k2_antithetic.yaml",
    "feniks_mode_frozen_ref_periodic_wake_k4.yaml",
    "feniks_mode_common15d_vem4_periodic_wake_k4.yaml",
)


@pytest.mark.parametrize("config_name", CONFIGS)
def test_mode_covering_configs_share_unsupervised_15d_contract(
    config_name: str,
) -> None:
    config = load_config(CONFIG_DIR / config_name)
    cfg = amortized_config(config)

    assert cfg["latent"]["normalization"] == "spline15d_checkpoint"
    assert cfg["latent"]["normalization_checkpoint"] == REFERENCE
    assert cfg["encoder"]["flow_output_space"] == "latent_x"
    assert not objective_uses_truth(cfg["objective"])


def test_mode_covering_configs_encode_the_four_control_cells() -> None:
    configs = {
        name: amortized_config(load_config(CONFIG_DIR / name)) for name in CONFIGS
    }

    common = configs["feniks_mode_common15d_vem4_elbo_k1.yaml"]
    antithetic = configs["feniks_mode_frozen_ref_elbo_k2_antithetic.yaml"]
    frozen_wake = configs["feniks_mode_frozen_ref_periodic_wake_k4.yaml"]
    learned_wake = configs["feniks_mode_common15d_vem4_periodic_wake_k4.yaml"]

    assert common["prior"]["source"] == "joint_realnvp"
    assert common["prior"]["update_schedule"] == "variational_em"
    assert antithetic["prior"]["source"] == "spline15d_checkpoint"
    assert antithetic["objective"]["sample_strategy"] == "antithetic"
    assert antithetic["training"]["n_samples"] == 2
    assert frozen_wake["objective"]["mode"] == "periodic_wake"
    assert frozen_wake["prior"]["train_jointly"] is False
    assert learned_wake["objective"]["mode"] == "periodic_wake"
    assert learned_wake["prior"]["train_jointly"] is True
