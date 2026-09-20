"""Submission plumbing only: scientific contract checks run in pipeline tests."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("running", [False, True])
def test_recovery_preserves_banks_and_replaces_only_blocked_chain(tmp_path, running):
    script = Path("scripts/resume_feniks_forward_population.sh").resolve()
    repo = tmp_path / "repo"
    root = tmp_path / "campaign"
    bin_dir = tmp_path / "bin"
    for folder in (repo, root, bin_dir):
        folder.mkdir()
    for folder in ("filters", "scripts", "configs", "euclid_dsps", "Data"):
        (repo / folder).mkdir()
    (repo / "pyproject.toml").touch()
    (root / "population").mkdir()
    (root / "population/best.eqx").write_bytes(b"saved classifier")
    (root / "CODE_DIR").write_text("original snapshot\n")
    (root / "CODE_SHA256").write_text("original digest\n")
    (root / "JOBS.env").write_text(
        "PREFLIGHT_JOB=10\nREFERENCE_JOB=11\nPOPULATION_JOB=12\n"
        "POSTERIOR_BANK_JOB=13\nTRAIN_JOB=14\nREPORT_JOB=15\n"
    )
    commands = {
        "squeue": "printf '13_[0-4%%4] PENDING\\n14 %s\\n15 PENDING\\n' "
        + ("RUNNING" if running else "PENDING"),
        # Mock read-only Python checks, not simulation or population fitting.
        "python": "printf '4\\n4\\n20\\n20\\n20\\n10\\n'",
        "scancel": 'printf "%s\\n" "$*" >> "$TRACE.cancel"',
        "sbatch": 'printf "%s %s\\n" "$FORWARD_MODE" "$*" >> "$TRACE.submit"\n'
        'count=$(wc -l < "$TRACE.submit")\nprintf "%s\\n" "$((100 + count))"',
    }
    for name, body in commands.items():
        path = bin_dir / name
        path.write_text("#!/bin/bash\nset -eu\n" + body + "\n")
        path.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        SCRATCH=str(tmp_path / "scratch"),
        TRACE=str(tmp_path / "trace"),
    )
    result = subprocess.run(
        ["bash", str(script), str(root)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert (root / "population/best.eqx").read_bytes() == b"saved classifier"
    if running:
        assert result.returncode != 0
        assert "Refusing recovery" in result.stderr
        assert not (tmp_path / "trace.cancel").exists()
        assert not (tmp_path / "trace.submit").exists()
        return
    assert result.returncode == 0, result.stderr
    assert set((tmp_path / "trace.cancel").read_text().split()) == {"13", "14", "15"}
    submitted = (tmp_path / "trace.submit").read_text().splitlines()
    assert [line.split()[0] for line in submitted] == [
        "population",
        "posterior-bank",
        "train",
        "report",
    ]
    assert "--dependency" not in submitted[0]
    assert "--array=0-4%4" in submitted[1]
    for line, dependency in zip(submitted[1:], (101, 102, 103), strict=True):
        assert f"--dependency=afterok:{dependency}" in line
    assert "REPORT_JOB=104" in (root / "JOBS.env").read_text()
    assert not (root / "RECOVERY_PENDING").exists()
    assert len(list((root / "recovery").glob("*/JOBS.previous.env"))) == 1
