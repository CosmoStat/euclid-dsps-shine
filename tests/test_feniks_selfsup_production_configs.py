import os
import subprocess
import sys
from pathlib import Path

import pytest

from euclid_dsps.amortized.config import amortized_config
from euclid_dsps.amortized.elbo import objective_mode, objective_uses_truth
from euclid_dsps.amortized.train import architecture_summary
from euclid_dsps.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
CONFIGS = (
    "feniks_selfsup_learned_rws_k8_t2.yaml",
    "feniks_selfsup_learned_rws_mix2_k8_t2.yaml",
    "feniks_selfsup_learned_smcwake_mix2_k4_t2.yaml",
)


@pytest.mark.parametrize("config_name", CONFIGS)
def test_production_configs_are_self_supervised_and_share_physics(
    config_name: str,
) -> None:
    cfg = amortized_config(load_config(CONFIG_DIR / config_name))

    assert cfg["latent"]["normalization"] == "spline15d_checkpoint"
    assert cfg["prior"]["source"] == "joint_realnvp"
    assert cfg["prior"]["train_jointly"] is True
    assert cfg["prior"]["init"] == "identity"
    assert cfg["encoder"]["flow_output_space"] == "latent_x"
    assert objective_mode(cfg["objective"]) == "reweighted_wake_sleep"
    assert not objective_uses_truth(cfg["objective"])
    assert cfg["likelihood"]["type"] == "student_t"
    assert cfg["likelihood"]["student_t_dof"] == 2.0
    assert cfg["likelihood"]["error_floor_frac"] == 0.0
    assert cfg["likelihood"]["error_jitter"] == 0.0
    assert cfg["objective"]["sleep"]["noise_family"] == "match_likelihood"
    assert cfg["objective"]["wake"]["train_prior"] is True


def test_production_configs_isolate_posterior_and_wake_changes() -> None:
    configs = {
        name: amortized_config(load_config(CONFIG_DIR / name)) for name in CONFIGS
    }
    baseline, mixture, smc = (configs[name] for name in CONFIGS)

    assert baseline["encoder"]["base_components"] == 1
    assert mixture["encoder"]["base_components"] == 2
    assert smc["encoder"]["base_components"] == 2
    assert baseline["objective"]["wake"]["sampler"] == "importance"
    assert mixture["objective"]["wake"]["sampler"] == "importance"
    assert smc["objective"]["wake"]["sampler"] == "smc"
    assert baseline["objective"]["wake"]["n_particles"] == 8
    assert mixture["objective"]["wake"]["n_particles"] == 8
    assert smc["objective"]["wake"]["n_particles"] == 4


@pytest.mark.parametrize("config_name", CONFIGS)
def test_checkpoint_architecture_records_mixture_and_noise_contract(
    config_name: str,
) -> None:
    summary = architecture_summary(load_config(CONFIG_DIR / config_name))
    expected_components = 1 if config_name == CONFIGS[0] else 2

    assert summary["encoder"]["base_components"] == expected_components
    assert summary["encoder"]["base_family"] == (
        "diag_gaussian" if expected_components == 1 else "mixture_diag_gaussian"
    )
    assert summary["objective"]["likelihood"] == "student_t"
    assert summary["objective"]["sleep"]["noise_family"] == "match_likelihood"


def test_production_preflight_does_not_import_jax() -> None:
    code = (
        "import sys; import validate_feniks_selfsup_production_inputs; "
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


def test_jacobian_lens_accepts_zero_byte_completion_marker() -> None:
    wrapper = (
        ROOT / "scripts" / "feniks_selfsup_production_jlens_h100.slurm"
    ).read_text()

    assert 'test -f "$TASK_DIR/DONE"' in wrapper
    assert 'test -s "$TASK_DIR/DONE"' not in wrapper


def test_production_wrapper_uses_compute_bounded_smc_epochs() -> None:
    wrapper = (ROOT / "scripts" / "feniks_selfsup_production_h100.slurm").read_text()

    assert "FULL_EPOCHS=(120 120 40)" in wrapper
    assert 'EPOCHS="${FULL_EPOCHS[$TASK_ID]}"' in wrapper


def test_smc_recovery_is_warm_restart_without_online_validation() -> None:
    wrapper = (
        ROOT / "scripts" / "feniks_selfsup_production_smc_resume_h100.slurm"
    ).read_text()
    submitter = (
        ROOT / "scripts" / "submit_feniks_selfsup_production_recovery.sh"
    ).read_text()

    assert 'START_EPOCH="${START_EPOCH:-40}"' in wrapper
    assert 'END_EPOCH="${END_EPOCH:-40}"' in wrapper
    assert '--initial-checkpoint "$INITIAL_CHECKPOINT"' in wrapper
    assert "--validation-every 0" in wrapper
    assert '"optimizer_state_resumed": false' in wrapper
    assert 'mv "$FAILED_TRAIN" "$ARCHIVE_TRAIN"' in submitter
    assert "start_epoch=$((source_epoch + 1))" in submitter
    assert "expected an epoch in [1, 39]" in submitter
    assert "START_EPOCH=$start_epoch,END_EPOCH=40" in submitter
    assert '--dependency="afterok:${resume_job}"' in submitter
    assert '--dependency="afterok:${lens_job}"' in submitter


@pytest.mark.parametrize(
    "wrapper_name",
    (
        "feniks_selfsup_production_h100.slurm",
        "feniks_selfsup_production_smc_resume_h100.slurm",
        "feniks_selfsup_production_jlens_h100.slurm",
        "feniks_selfsup_production_finalize_h100.slurm",
    ),
)
def test_production_wrappers_use_node_local_matplotlib_cache(
    wrapper_name: str,
) -> None:
    wrapper = (ROOT / "scripts" / wrapper_name).read_text()

    assert "JOBSCRATCH" not in wrapper
    assert 'export MPLCONFIGDIR="/tmp/mpl-' in wrapper
