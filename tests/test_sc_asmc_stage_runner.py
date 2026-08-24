from __future__ import annotations

import subprocess
import sys


def test_stage_runner_help_exposes_resumeable_phases() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_feniks_sc_asmc_em.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for phase in (
        "prepare",
        "sleep",
        "smoke",
        "preflight",
        "estep",
        "merge-estep",
        "prior-mstep",
        "reweight",
        "merge-reweight",
        "q-distill",
        "report",
        "validate",
        "mark-training-complete",
        "run-single-node",
        "status",
    ):
        assert phase in result.stdout
