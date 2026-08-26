from __future__ import annotations

import subprocess
from pathlib import Path


def test_sixteen_gpu_launcher_uses_four_independent_four_gpu_shards() -> None:
    launcher = Path("scripts/feniks_sc_asmc_em_16gpu.slurm").read_text(encoding="utf-8")
    worker = Path("scripts/feniks_sc_asmc_em_array_worker.sh").read_text(
        encoding="utf-8"
    )
    validator = Path("euclid_dsps/amortized/sc_asmc_validate.py").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --array=0-3%4" in launcher
    assert "#SBATCH --gres=gpu:4" in launcher
    assert "SHARD_COUNT=4" in launcher
    assert "--expected-devices 4" in worker
    assert 'estep --iteration 1 --shard-id "$TASK_ID"' in worker
    assert 'estep --iteration 2 --shard-id "$TASK_ID"' in worker
    assert 'reweight --iteration 1 --shard-id "$TASK_ID"' in worker
    assert 'reweight --iteration 2 --shard-id "$TASK_ID"' in worker
    assert "SMOKE_ROOT" in worker
    assert "validate_feniks_sc_asmc_smoke.py" in worker
    assert "REQUIRE_SMOKE16=1" in launcher
    assert 'SMOKE16_ROOT/shard_$smoke_shard' in worker
    assert "SMOKE_PASS" in validator
    assert "smoke configuration differs" in validator
    assert "smoke dataset differs" in validator
    assert "smoke code commit differs" in validator
    assert "jax.distributed" not in worker
    assert "srun" not in worker


def test_four_eight_and_sixteen_gpu_launchers_have_consistent_shard_counts() -> None:
    expected = {
        "feniks_sc_asmc_em_4gpu.slurm": (1, None),
        "feniks_sc_asmc_em_8gpu.slurm": (2, "#SBATCH --array=0-1%2"),
        "feniks_sc_asmc_em_16gpu.slurm": (4, "#SBATCH --array=0-3%4"),
    }
    for name, (shards, array_line) in expected.items():
        value = Path("scripts", name).read_text(encoding="utf-8")
        assert f"SHARD_COUNT={shards}" in value
        assert "#SBATCH --gres=gpu:4" in value
        if array_line is not None:
            assert array_line in value


def test_sixteen_gpu_smoke_uses_disjoint_four_gpu_shards() -> None:
    launcher = Path("scripts/feniks_sc_asmc_em_16gpu_smoke.slurm").read_text(
        encoding="utf-8"
    )
    four_gpu = Path("scripts/feniks_sc_asmc_em_4gpu_smoke.slurm").read_text(
        encoding="utf-8"
    )
    runner = Path("scripts/run_feniks_sc_asmc_em.py").read_text(encoding="utf-8")

    assert "#SBATCH --array=0-3%4" in launcher
    assert "#SBATCH --gres=gpu:4" in launcher
    assert 'RUN_ROOT="$SMOKE16_ROOT/shard_$TASK_ID"' in launcher
    assert 'SMOKE_ROW_OFFSET="$((TASK_ID * 8))"' in launcher
    assert 'UPSTREAM_SMOKE_ROOT="$SMOKE4_ROOT"' in launcher
    assert '--smoke-root "$UPSTREAM_SMOKE_ROOT"' in four_gpu
    assert '--smoke-root "$SMOKE4_ROOT"' not in four_gpu
    assert "--row-offset" in runner


def test_sc_asmc_shell_launchers_pass_bash_static_validation() -> None:
    paths = [
        "scripts/feniks_sc_asmc_em_array_worker.sh",
        "scripts/feniks_sc_asmc_em_4gpu.slurm",
        "scripts/feniks_sc_asmc_em_8gpu.slurm",
        "scripts/feniks_sc_asmc_em_16gpu.slurm",
        "scripts/feniks_sc_asmc_em_4gpu_smoke.slurm",
        "scripts/feniks_sc_asmc_em_16gpu_smoke.slurm",
        "scripts/feniks_sc_asmc_postfreeze_nuts_8gpu.slurm",
        "scripts/feniks_sc_asmc_report_resume_4gpu.slurm",
        "scripts/feniks_sc_asmc_repair_report_4gpu.slurm",
    ]
    subprocess.run(["bash", "-n", *paths], check=True)


def test_report_resume_launcher_requires_frozen_training_and_bounds_memory() -> None:
    launcher = Path("scripts/feniks_sc_asmc_report_resume_4gpu.slurm").read_text(
        encoding="utf-8"
    )

    assert 'test -s "$RUN_ROOT/TRAINING_COMPLETE.json"' in launcher
    assert 'test -s "$RUN_ROOT/banks/em2_p2/posterior_bank_manifest.json"' in launcher
    assert 'XLA_PYTHON_CLIENT_MEM_FRACTION:-0.72' in launcher
    assert '"${COMMON[@]}" report' in launcher
    assert '"${COMMON[@]}" validate' in launcher
    assert '"${COMMON[@]}" status' in launcher


def test_final_repair_launcher_retries_only_post_em_inference() -> None:
    launcher = Path("scripts/feniks_sc_asmc_repair_report_4gpu.slurm").read_text(
        encoding="utf-8"
    )

    assert 'test -s "$RUN_ROOT/TRAINING_COMPLETE.json"' in launcher
    assert 'repair-final --shard-id 0 --shard-count 1' in launcher
    assert 'merge-repair-final --shard-count 1' in launcher
    assert '"${COMMON[@]}" report' in launcher
    assert '"${COMMON[@]}" validate' in launcher
    assert "prior-mstep" not in launcher
    assert "q-distill" not in launcher


def test_postfreeze_nuts_is_absent_from_training_worker() -> None:
    worker = Path("scripts/feniks_sc_asmc_em_array_worker.sh").read_text(
        encoding="utf-8"
    )
    nuts = Path("scripts/feniks_sc_asmc_postfreeze_nuts_8gpu.slurm").read_text(
        encoding="utf-8"
    )

    assert "nuts" not in worker.lower()
    assert "#SBATCH --array=0-7%8" in nuts
    assert "validate_postfreeze_gate" in nuts
    assert "--ignore-truth" in nuts
