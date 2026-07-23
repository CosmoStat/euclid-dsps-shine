from __future__ import annotations

from pathlib import Path

import pytest

from euclid_dsps.amortized.config import amortized_config
from euclid_dsps.amortized.elbo import objective_mode, objective_uses_truth
from euclid_dsps.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
CONFIGS = (
    "feniks_selfsup_paper_rws_k8_t2_seed2.yaml",
    "feniks_selfsup_paper_rws_k8_t2_seed3.yaml",
    "feniks_selfsup_paper_fixed_prior_rws_k8_t2.yaml",
    "feniks_selfsup_paper_avi_joint_t2.yaml",
)


@pytest.mark.parametrize("config_name", CONFIGS)
def test_paper_configs_share_physics_and_exclude_truth(config_name: str) -> None:
    cfg = amortized_config(load_config(CONFIG_DIR / config_name))

    assert cfg["latent"]["normalization"] == "spline15d_checkpoint"
    assert cfg["encoder"]["type"] == "conditional_flow"
    assert cfg["encoder"]["flow_family"] == "realnvp"
    assert cfg["encoder"]["flow_output_space"] == "latent_x"
    assert cfg["likelihood"]["type"] == "student_t"
    assert cfg["likelihood"]["student_t_dof"] == 2.0
    assert cfg["likelihood"]["error_floor_frac"] == 0.0
    assert cfg["likelihood"]["error_jitter"] == 0.0
    assert not objective_uses_truth(cfg["objective"])


def test_paper_matrix_has_the_intended_objective_and_prior_controls() -> None:
    configs = [
        amortized_config(load_config(CONFIG_DIR / name)) for name in CONFIGS
    ]

    assert [objective_mode(cfg["objective"]) for cfg in configs] == [
        "reweighted_wake_sleep",
        "reweighted_wake_sleep",
        "reweighted_wake_sleep",
        "stochastic_elbo",
    ]
    assert [cfg["prior"]["source"] for cfg in configs] == [
        "joint_realnvp",
        "joint_realnvp",
        "spline15d_checkpoint",
        "joint_realnvp",
    ]
    assert configs[0]["training"]["seed"] != configs[1]["training"]["seed"]
    assert configs[2]["prior"]["train_jointly"] is False
    assert configs[2]["objective"]["wake"]["train_prior"] is False


def test_paper_wrappers_encode_four_training_and_sixteen_lens_tasks() -> None:
    train = (ROOT / "scripts" / "feniks_selfsup_paper_h100.slurm").read_text()
    lens = (ROOT / "scripts" / "feniks_selfsup_paper_jlens_h100.slurm").read_text()
    finalize = (
        ROOT / "scripts" / "feniks_selfsup_paper_finalize_h100.slurm"
    ).read_text()

    assert "#SBATCH --array=0-3%4" in train
    assert "#SBATCH --array=0-15%16" in lens
    assert "#SBATCH --array=0-3%4" in finalize
    assert '--posterior-samples "$POSTERIOR_SAMPLES"' in lens
    assert "JOBSCRATCH" not in train + lens + finalize
    assert 'test -f "$TASK_DIR/DONE"' in lens
