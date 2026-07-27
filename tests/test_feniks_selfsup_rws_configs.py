import os
import subprocess
import sys
from pathlib import Path

import pytest

from euclid_dsps.amortized.config import amortized_config
from euclid_dsps.amortized.elbo import objective_mode, objective_uses_truth
from euclid_dsps.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
REFERENCE = "outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx"

CONFIGS = (
    "feniks_selfsup_fixed_ref_rws_k4_gaussian.yaml",
    "feniks_selfsup_learned_rws_k4_gaussian.yaml",
    "feniks_selfsup_learned_rws_sleep3_wake1_k4.yaml",
    "feniks_selfsup_learned_rws_sleep3_wake1_k8.yaml",
)


@pytest.mark.parametrize("config_name", CONFIGS)
def test_selfsup_rws_configs_share_physics_and_normalization(config_name: str) -> None:
    config = load_config(CONFIG_DIR / config_name)
    cfg = amortized_config(config)

    assert cfg["latent"]["normalization"] == "spline15d_checkpoint"
    assert cfg["latent"]["normalization_checkpoint"] == REFERENCE
    assert cfg["encoder"]["flow_output_space"] == "latent_x"
    assert objective_mode(cfg["objective"]) == "reweighted_wake_sleep"
    assert not objective_uses_truth(cfg["objective"])
    assert cfg["objective"]["sleep"]["enabled"] is True
    assert cfg["likelihood"]["type"] == "gaussian"
    assert cfg["likelihood"]["error_floor_frac"] == 0.0
    assert cfg["likelihood"]["error_jitter"] == 0.0
    assert cfg["training"]["validation_every"] == 8
    assert cfg["training_snapshots"]["enabled"] is False


def test_selfsup_rws_configs_encode_control_and_three_learned_priors() -> None:
    configs = {
        name: amortized_config(load_config(CONFIG_DIR / name)) for name in CONFIGS
    }
    frozen = configs[CONFIGS[0]]
    wake_only = configs[CONFIGS[1]]
    sleep_k4 = configs[CONFIGS[2]]
    sleep_k8 = configs[CONFIGS[3]]

    assert frozen["prior"]["source"] == "spline15d_checkpoint"
    assert frozen["prior"]["train_jointly"] is False
    assert frozen["objective"]["wake"]["train_prior"] is False
    for cfg in (wake_only, sleep_k4, sleep_k8):
        assert cfg["prior"]["source"] == "joint_realnvp"
        assert cfg["prior"]["train_jointly"] is True
        assert cfg["prior"]["init"] == "identity"
        assert cfg["prior"]["update_schedule"] == "joint"
        assert cfg["objective"]["wake"]["train_prior"] is True
    assert wake_only["objective"]["wake"]["every_encoder_epochs"] == 1
    assert sleep_k4["objective"]["wake"]["n_particles"] == 4
    assert sleep_k8["objective"]["wake"]["n_particles"] == 8


def test_selfsup_rws_preflight_does_not_import_jax() -> None:
    code = (
        "import sys; "
        "import validate_feniks_selfsup_rws_inputs; "
        "assert not any(name == 'jax' or name.startswith('jax.') "
        "for name in sys.modules)"
    )
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cuda"
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "scripts"), str(ROOT)))
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
